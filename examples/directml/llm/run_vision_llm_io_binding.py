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
from model_type_mapping import get_model_dir, get_supported_lvlm_models
from PIL import Image
from transformers import AutoProcessor


def run_vision_llm_io_binding(
    model_type: str,
    prompt: str,
    image_path: str,
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
    llm_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "decoder_model_merged.onnx"),
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16
    num_layers = 0
    num_key_value_heads = 0
    head_dim = []
    for inputs_meta in llm_session._inputs_meta:  # pylint: disable=protected-access
        if inputs_meta.name.startswith("past_key_values.") and inputs_meta.name.endswith(".key"):
            num_layers += 1
            num_key_value_heads = inputs_meta.shape[1]
            head_dim = inputs_meta.shape[3]

    # Initialize the tokenizer and produce the initial tokens.
    processor = AutoProcessor.from_pretrained(model_dir)
    image = Image.open(image_path)

    processed_prompt = f"USER: {prompt}\n<image>\nASSISTANT:"
    processed_inputs = processor(text=processed_prompt, images=image, return_tensors="np")
    input_ids = processed_inputs["input_ids"]
    pixel_values = processed_inputs["pixel_values"].astype(np.float16)

    input_ids = np.asarray(input_ids, dtype=np.int64)
    input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(input_ids, device)
    input_ids_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, device)
    sequence_length = input_ids.shape()[1]

    attention_mask = np.zeros((1, max_seq_len), dtype=np.int64)
    attention_mask[:, :sequence_length] = 1

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

    llm_io_binding.bind_cpu_input("pixel_values", pixel_values)
    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))
    llm_io_binding.bind_output("logits", device)
    llm_io_binding.bind_ortvalue_input("input_ids", input_ids)
    llm_io_binding.bind_ortvalue_input("input_ids_increment", input_ids_increment)

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        llm_io_binding.bind_cpu_input("attention_mask", attention_mask)

        for layer_idx in range(num_layers):
            llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.value", v_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.value", v_caches[layer_idx])

        # Run the LLM
        llm_session.run_with_iobinding(llm_io_binding)

        # Decide the next token using your preferred sampling strategy.
        logits = llm_io_binding.get_outputs()[0].numpy()
        last_token_logits = logits[:, -1, :]
        next_token = np.argmax(last_token_logits, axis=-1, keepdims=True)
        output_tokens.append(next_token.item())

        # Set the token for the next iteration
        llm_io_binding.bind_cpu_input("input_ids_increment", next_token)

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and output_tokens[-1] == processor.tokenizer.eos_token_id:
            break

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
            llm_io_binding.bind_output("logits", device)
            sequence_length = logits.shape[1]
            attention_mask = np.zeros((1, max_seq_len), dtype=np.int64)
            attention_mask[:, :sequence_length] = 1

        if sequence_length < max_seq_len:
            attention_mask[:, sequence_length] = 1
            sequence_length += 1

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration

    # Only print the tokens/s when ignore_eos is provided for benchmarking purposes
    if ignore_eos:
        print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    output_str = processor.decode(output_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=False)

    print(output_str)


if __name__ == "__main__":
    default_image_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "placeholder.png")

    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is in this image?")
    parser.add_argument("--image", type=str, default=default_image_path)
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--device", type=str, choices=["dml", "cuda"], default="dml")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--model_type",
        choices=["local", *get_supported_lvlm_models()],
        help="Which model to run.",
        required=True,
        type=str,
    )

    args = parser.parse_args()
    run_vision_llm_io_binding(
        args.model_type,
        args.prompt,
        args.image,
        args.max_seq_len,
        args.max_gen_len,
        args.device,
        args.device_id,
        args.ignore_eos,
    )
