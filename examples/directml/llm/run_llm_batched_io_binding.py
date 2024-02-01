# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# This program will run the ONNX version of the LLM.
import argparse
import os
import time
from typing import List

import numpy as np
import onnxruntime
from transformers import AutoTokenizer
from chat_templates import get_chat_template
from model_type_mapping import get_supported_llm_models, get_model_dir


def run_llm_io_binding(
    model_type: str,
    prompts: List[str],
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    device: str = "dml",
    device_id: int = 0,
    ignore_eos: bool = False,
) -> str:
    onnxruntime.set_default_logger_severity(3)

    execution_provider = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }[device]

    # Create the ONNX session
    providers = [
        (
            execution_provider,
            {
                "device_id": device_id,
            },
        )
    ]

    model_dir = get_model_dir(model_type)
    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_free_dimension_override_by_name("batch_size", len(prompts))
    llm_session_options.add_free_dimension_override_by_name("max_seq_len", max_seq_len)
    llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)
    llm_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "decoder_model_merged.onnx"),
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16
    num_layers = 0
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name.startswith("cache.") and inputs_meta.name.endswith(".key"):
            num_layers += 1
            num_key_value_heads = inputs_meta.shape[1]
            head_dim = inputs_meta.shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = get_chat_template(model_type) or tokenizer.chat_template

    batch_size = len(prompts)

    batched_tokens = []
    seq_lens = []
    past_seq_lens = [0] * batch_size
    max_tokens_len = 0
    for prompt in prompts:
        # Generate the tokens
        prompt_tokens = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="np")
        prompt_tokens = np.asarray(prompt_tokens, dtype=np.int64)
        batched_tokens.append(prompt_tokens)

        # Generate the mask
        token_count = prompt_tokens.shape[1]
        seq_lens.append(token_count)

    max_tokens_len = max(seq_lens)

    # Pad the leading missing tokens with 0
    for idx in range(len(batched_tokens)):
        batched_tokens[idx] = np.pad(batched_tokens[idx], ((0, 0), (0, max_tokens_len - batched_tokens[idx].shape[1])))

    tokens = np.concatenate(batched_tokens, axis=0)

    # When we reach this point, tokens is an array of shape [batch_size, max_tokens_len] that looks like this:
    #
    # [ 0,    0,   0, 131,  15]
    # [ 0,   37,  94,  16,  20]
    # [65, 5341, 894, 365,  24]
    # [ 0,    0, 524,  25, 124]
    #
    # Where 0 represents padding that was added for prompts that were shorter, which will be masked out in the model
    tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, device)
    tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((batch_size, 1), np.int64, device)

    # Create the LLM model's I/O binding
    logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        (batch_size, max_tokens_len, tokenizer.vocab_size), data_type, device
    )
    llm_io_binding = llm_session.io_binding()
    llm_io_binding.bind_ortvalue_output("logits", logits)

    # Create the K and V caches.
    cache_shape = (batch_size, num_key_value_heads, max_seq_len, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_caches = []
    v_caches = []
    k_caches_out = []
    v_caches_out = []

    for _ in range(num_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))
        k_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, device)
        )
        v_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, device)
        )

    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))
    llm_io_binding.bind_output("logits", device)
    llm_io_binding.bind_ortvalue_input("tokens", tokens)
    llm_io_binding.bind_ortvalue_input("tokens_increment", tokens_increment)

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    batched_output_tokens = []
    for _ in range(batch_size):
        batched_output_tokens.append([])

    eos_found = [False] * batch_size
    eos_count = 0
    for idx in range(max_gen_len):
        if idx == 0:
            position_ids = np.arange(max_tokens_len, dtype=np.int64).reshape((1, max_tokens_len))
            position_ids = np.broadcast_to(position_ids, (batch_size, max_tokens_len))
            llm_io_binding.bind_cpu_input("position_ids", position_ids)
        else:
            position_ids_increment = np.array(seq_lens, dtype=np.int64).reshape((batch_size, 1))
            llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

        seqlens_k = np.array(past_seq_lens, dtype=np.int32, ndmin=1)
        llm_io_binding.bind_cpu_input("seqlens_k", seqlens_k)

        for layer_idx in range(num_layers):
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", v_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", v_caches[layer_idx])

        # Run the LLM
        llm_session.run_with_iobinding(llm_io_binding)

        # Decide the next token using your preferred sampling strategy.
        if idx == 0:
            logits = np.take_along_axis(
                llm_io_binding.get_outputs()[0].numpy(), np.array(seq_lens).reshape(batch_size, 1, 1) - 1, axis=1
            )
        else:
            logits = llm_io_binding.get_outputs()[0].numpy()[:, -1, :]

        next_tokens = np.argmax(logits, axis=-1, keepdims=False).reshape(batch_size, 1)

        # Set the token for the next iteration
        llm_io_binding.bind_cpu_input("tokens_increment", next_tokens)

        tokens_list = next_tokens.tolist()
        for output_token_idx in range(len(tokens_list)):
            output_token = tokens_list[output_token_idx][0]

            # print(output_token)
            if not eos_found[output_token_idx] and output_token == tokenizer.eos_token_id:
                eos_found[output_token_idx] = True
                eos_count += 1

            if not eos_found[output_token_idx]:
                batched_output_tokens[output_token_idx].append(output_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and eos_count == batch_size:
            break

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
            llm_io_binding.bind_output("logits", device)

        for seq_len_idx in range(len(seq_lens)):
            past_seq_lens[seq_len_idx] = seq_lens[seq_len_idx]
            seq_lens[seq_len_idx] += 1

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration

    # Only print the tokens/s when ignore_eos is provided for benchmarking purposes
    if ignore_eos:
        print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    for prompt_idx in range(len(prompts)):
        print("")
        print("")
        print(f"Prompt {prompt_idx} > {prompts[prompt_idx]}")
        print("")
        output_str = tokenizer.decode(batched_output_tokens[prompt_idx])
        print(f"Answer {prompt_idx} > {output_str}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prompts",
        type=str,
        nargs="+",
        default=["What is the lightest element?", "What is the difference between nuclear fission and nuclear fusion?"],
    )
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--model_type",
        choices=get_supported_llm_models(),
        help="Which model to run.",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    run_llm_io_binding(
        args.model_type,
        args.prompts,
        args.max_seq_len,
        args.max_gen_len,
        args.device,
        args.device_id,
        args.ignore_eos,
    )
