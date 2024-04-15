# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

# This program will run the ONNX version of the LLM.
import argparse
import os
import time

import numpy as np
import onnxruntime
from chat_templates import get_chat_template
from model_type_mapping import get_model_dir, get_supported_llm_models
from transformers import AutoTokenizer


def run_llm_io_binding(
    model_type: str,
    prompt: str,
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

    # Initialize the providers
    providers = [
        (
            execution_provider,
            {
                "device_id": device_id,
            },
        )
    ]

    if device == "cuda":
        providers[0][1]["enable_cuda_graph"] = True

    model_dir = get_model_dir(model_type)
    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_session_config_entry("ep.dml.enable_graph_capture", "1")

    llm_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "model.onnx"),
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16
    num_layers = 0
    for inputs_meta in llm_session._inputs_meta:  # pylint: disable=protected-access
        if inputs_meta.name.startswith("past_key_values.") and inputs_meta.name.endswith(".key"):
            num_layers += 1
            num_key_value_heads = inputs_meta.shape[1]
            head_dim = inputs_meta.shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    tokenizer.chat_template = get_chat_template(model_type) or tokenizer.chat_template

    initial_input_ids = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], return_tensors="np")
    initial_input_ids = np.asarray(initial_input_ids, dtype=np.int64)
    initial_input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(initial_input_ids, device)

    position_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)
    attention_mask_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len), np.int64, device)
    input_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)

    sequence_length = initial_input_ids.shape()[1]

    # Create the LLM model's I/O binding
    llm_io_binding = llm_session.io_binding()

    # Create the K and V caches.
    cache_shape = (1, num_key_value_heads, max_seq_len, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_caches = []
    v_caches = []

    for _ in range(num_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, device))

    llm_io_binding.bind_output("logits", device)
    llm_io_binding.bind_ortvalue_input("input_ids", initial_input_ids)

    initial_position_ids = np.arange(sequence_length, dtype=np.int64).reshape((1, sequence_length))
    llm_io_binding.bind_cpu_input("position_ids", initial_position_ids)

    for layer_idx in range(num_layers):
        llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.key", k_caches[layer_idx])
        llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.value", v_caches[layer_idx])
        llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.key", k_caches[layer_idx])
        llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.value", v_caches[layer_idx])

    attention_mask = np.pad(np.ones((1, sequence_length), dtype=np.int64), ((0, 0), (0, max_seq_len - sequence_length)))
    llm_io_binding.bind_ortvalue_input("attention_mask", attention_mask_ortvalue)

    run_options = onnxruntime.RunOptions()

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        if idx > 0:
            position_ids = np.array(sequence_length - 1, dtype=np.int64, ndmin=2)
            position_ids_ortvalue.update_inplace(position_ids)

        if idx == 1:
            llm_io_binding.bind_ortvalue_input("position_ids", position_ids_ortvalue)
            llm_io_binding.bind_ortvalue_input("input_ids", input_ids_ortvalue)

        attention_mask[0, sequence_length - 1] = 1
        attention_mask_ortvalue.update_inplace(attention_mask)

        # Run the LLM
        if idx == 0:
            run_options.add_run_config_entry("gpu_graph_id", "-1")
        elif idx == 1:
            run_options.add_run_config_entry("gpu_graph_id", "1")

        llm_session.run_with_iobinding(llm_io_binding, run_options)

        # Decide the next token using your preferred sampling strategy.
        logits = llm_io_binding.get_outputs()[0].numpy()[:, -1, :]
        next_token = np.argmax(logits, axis=-1, keepdims=True)
        output_tokens.append(next_token.item())

        # Set the token for the next iteration
        input_ids_ortvalue.update_inplace(next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and output_tokens[-1] == tokenizer.eos_token_id:
            break

        if idx == 0:
            llm_io_binding.bind_output("logits", device)

        sequence_length += 1
        sequence_length = min(sequence_length, max_seq_len)

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration

    # Only print the tokens/s when ignore_eos is provided for benchmarking purposes
    if ignore_eos:
        print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    output_str = tokenizer.decode(output_tokens, skip_special_tokens=True)

    print(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the lightest element?")
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
        args.prompt,
        args.max_seq_len,
        args.max_gen_len,
        args.device,
        args.device_id,
        args.ignore_eos,
    )
