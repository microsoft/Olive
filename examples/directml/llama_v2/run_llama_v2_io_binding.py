# This program will run the ONNX version of the LlamaV2 model.
import argparse
import os
import time
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
from sentencepiece import SentencePieceProcessor

# ruff: noqa: T201


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert Path(model_path).is_file(), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert isinstance(s, str)
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id, *t]
        if eos:
            t = [*t, self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


def run_llama_v2_io_binding(
    prompt: str,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    device_id: int = 0,
    disable_metacommands: bool = False,
    ignore_eos: bool = False,
    model_dir: str = "models/optimized/llama_v2",
) -> str:
    onnxruntime.set_default_logger_severity(3)

    # Create the ONNX session
    providers = [
        (
            "DmlExecutionProvider",
            {
                "disable_metacommands": disable_metacommands,
                "device_id": device_id,
            },
        )
    ]

    sampling_session_options = onnxruntime.SessionOptions()
    sampling_session_options.add_free_dimension_override_by_name("batch_size", 1)
    argmax_sampling_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "argmax_sampling/model.onnx"),
        sess_options=sampling_session_options,
        providers=providers,
    )

    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
    llm_session_options.add_free_dimension_override_by_name("max_seq_len", max_seq_len)
    llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)
    llm_session = onnxruntime.InferenceSession(
        os.path.join(model_dir, "llama_v2/decoder_model_merged.onnx"),
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16

    hidden_size = 4096
    n_heads = 32
    n_layers = 0

    for inputs_meta in llm_session._inputs_meta:  # pylint: disable=protected-access
        if inputs_meta.name.startswith("past_key_values.") and inputs_meta.name.endswith(".key"):
            n_layers += 1

    binding_device = "dml"

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path=os.path.join(model_dir, "tokenizer.model"))
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = np.expand_dims(np.asarray(tokens, dtype=np.int64), 0)
    tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, binding_device)
    tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, binding_device)

    seq_len = tokens.shape()[1]

    # Create the attention mask, which contains 1's for values that should stay intact, and 0's for values that should
    # get added to -10000
    attn_mask = np.pad(np.ones((1, seq_len)), ((0, 0), (max_seq_len - seq_len, 0))).astype(np.int32)
    attn_mask = onnxruntime.OrtValue.ortvalue_from_numpy(attn_mask, binding_device)
    attn_mask_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len), np.int32, binding_device)

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)

    # Create the argmax sampling's I/O binding
    argmax_sampling_io_binding = argmax_sampling_session.io_binding()
    argmax_sampling_io_binding.bind_ortvalue_output("next_token", tokens_increment)

    # Create the LLM model's I/O binding
    logits_shape = (1, tokenizer.n_words)
    logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(logits_shape, data_type, binding_device)
    llm_io_binding = llm_session.io_binding()
    llm_io_binding.bind_ortvalue_output("logits", logits)

    cache_shape = (1, n_heads, max_seq_len, head_dim)
    initial_cache = np.zeros(cache_shape, dtype=data_type)
    k_caches = []
    v_caches = []
    k_caches_out = []
    v_caches_out = []

    for _ in range(n_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device))
        k_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, binding_device)
        )
        v_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, binding_device)
        )

    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        if idx == 0:
            position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
            llm_io_binding.bind_cpu_input("position_ids", position_ids)
        else:
            position_ids_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))
            llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

        # Run the LLM model
        llm_io_binding.bind_ortvalue_input("tokens", tokens)
        llm_io_binding.bind_ortvalue_input("tokens_increment", tokens_increment)
        llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
        llm_io_binding.bind_ortvalue_output("attn_mask_out", attn_mask_out)

        for layer_idx in range(n_layers):
            llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.value", v_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.key", k_caches_out[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.value", v_caches_out[layer_idx])

        llm_session.run_with_iobinding(llm_io_binding)
        llm_io_binding.synchronize_outputs()

        # Decide the next token using your preferred sampling strategy.
        argmax_sampling_io_binding.bind_ortvalue_input("logits", logits)
        argmax_sampling_session.run_with_iobinding(argmax_sampling_io_binding)
        argmax_sampling_io_binding.synchronize_outputs()
        output_tokens.append(tokens_increment.numpy().item())

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if not ignore_eos and output_tokens[-1] == tokenizer.eos_id:
            break

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))

        attn_mask_out, attn_mask = attn_mask, attn_mask_out
        k_caches, k_caches_out = k_caches_out, k_caches
        v_caches, v_caches_out = v_caches_out, v_caches

        seq_len += 1

    after_time = time.perf_counter()
    duration = after_time - before_time
    tokens_per_second = idx / duration
    print(f"Execution took {duration:0.4f} seconds (generated {tokens_per_second:0.2f} tokens per second)")

    output_str = tokenizer.decode(output_tokens)

    print(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the lightest element?")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--disable_metacommands", action="store_true")
    parser.add_argument("--ignore_eos", action="store_true")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument(
        "--model_dir",
        type=str,
        default="models/optimized/llama_v2",
        help="Path to the folder containing the argmax_sampling folder, llama_v2 folder and the tokenizer.model file",
    )

    args = parser.parse_args()
    run_llama_v2_io_binding(
        args.prompt,
        args.max_seq_len,
        args.max_gen_len,
        args.device_id,
        args.disable_metacommands,
        args.ignore_eos,
        args.model_dir,
    )
