# This program will run the ONNX version of the LlamaV2 model.
import argparse
import os
import time
from typing import List

import numpy as np
import onnxruntime
from sentencepiece import SentencePieceProcessor


class Tokenizer:
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = SentencePieceProcessor(model_file=model_path)

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.vocab_size()
        self.bos_id: int = self.sp_model.bos_id()
        self.eos_id: int = self.sp_model.eos_id()
        self.pad_id: int = self.sp_model.pad_id()

        assert self.sp_model.vocab_size() == self.sp_model.get_piece_size()

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s)
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)


def run_llama_v2_io_binding(
    prompt: str,
    max_seq_len: int = 2048,
    max_gen_len: int = 256,
    disable_metacommands: bool = False,
) -> str:
    onnxruntime.set_default_logger_severity(3)

    # Create the ONNX session
    providers = [
        (
            "DmlExecutionProvider",
            {
                "disable_metacommands": disable_metacommands,
            },
        )
    ]

    update_embeddings_session = onnxruntime.InferenceSession(
        "models/optimized/llama_v2/update_embeddings/model.onnx",
        sess_options=onnxruntime.SessionOptions(),
        providers=providers,
    )

    argmax_sampling_session = onnxruntime.InferenceSession(
        "models/optimized/llama_v2/argmax_sampling/model.onnx",
        sess_options=onnxruntime.SessionOptions(),
        providers=providers,
    )

    llm_session_options = onnxruntime.SessionOptions()
    llm_session_options.add_free_dimension_override_by_name("max_seq_len", max_seq_len)
    llm_session = onnxruntime.InferenceSession(
        "models/optimized/llama_v2/llama_v2/decoder_model_merged.onnx",
        sess_options=llm_session_options,
        providers=providers,
    )

    data_type = np.float16

    # Get the relevant shapes so we can create the inputs
    for inputs_meta in llm_session._inputs_meta:
        if inputs_meta.name == "x":
            x_shape = inputs_meta.shape
        elif inputs_meta.name == "attn_mask":
            attn_mask_shape = inputs_meta.shape
        elif inputs_meta.name == "cache.0.key":
            cache_shape = inputs_meta.shape

    n_layers = 32
    hidden_size = x_shape[2]
    n_heads = cache_shape[1]
    attn_mask_shape[1] = max_seq_len
    attn_mask_shape[2] = max_seq_len

    binding_device = "dml"

    # Initialize the tokenizer and produce the initial tokens.
    tokenizer = Tokenizer(model_path="models/optimized/llama_v2/tokenizer.model")
    tokens = tokenizer.encode(prompt, bos=True, eos=False)
    tokens = onnxruntime.OrtValue.ortvalue_from_numpy(np.asarray(tokens, dtype=np.int64), binding_device)

    # Create the attention mask, which contains 1's for values that should stay intact, and 0's for values that should
    # get added to -10000
    attn_mask = np.tril(np.ones(attn_mask_shape)).astype(np.int32)
    attn_mask = onnxruntime.OrtValue.ortvalue_from_numpy(attn_mask, binding_device)
    attn_mask_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(attn_mask_shape, np.int32, binding_device)

    # Create the K and V caches.
    head_dim = int(hidden_size / n_heads)

    # Create the argmax sampling's I/O binding
    next_token = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1,), np.int64, binding_device)
    argmax_sampling_io_binding = argmax_sampling_session.io_binding()
    argmax_sampling_io_binding.bind_ortvalue_output("next_token", next_token)

    x = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
        (1, tokens.shape()[0], hidden_size), data_type, binding_device
    )
    x_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1, hidden_size), data_type, binding_device)
    cos = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)
    cos_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)
    sin = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)
    sin_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, max_seq_len, 1, 64), data_type, binding_device)

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

    for layer_idx in range(n_layers):
        k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device))
        v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, binding_device))
        k_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, binding_device)
        )
        v_caches_out.append(
            onnxruntime.OrtValue.ortvalue_from_shape_and_type(initial_cache.shape, initial_cache.dtype, binding_device)
        )

    llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))

    update_embeddings_io_binding = update_embeddings_session.io_binding()
    update_embeddings_io_binding.bind_ortvalue_input("tokens", tokens)
    update_embeddings_io_binding.bind_ortvalue_output("embeddings", x)

    before_time = time.perf_counter()

    # Iteratively generate tokens.
    output_tokens = []
    for idx in range(max_gen_len):
        # Update the embeddings
        update_embeddings_session.run_with_iobinding(update_embeddings_io_binding)
        update_embeddings_io_binding.synchronize_outputs()

        # Run the LLM model
        llm_io_binding.bind_ortvalue_input("x", x)
        llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
        llm_io_binding.bind_ortvalue_input("x_increment", x_increment)
        llm_io_binding.bind_ortvalue_input("cos", cos)
        llm_io_binding.bind_ortvalue_input("sin", sin)
        llm_io_binding.bind_ortvalue_output("attn_mask_out", attn_mask_out)
        llm_io_binding.bind_ortvalue_output("cos_out", cos_out)
        llm_io_binding.bind_ortvalue_output("sin_out", sin_out)

        for layer_idx in range(n_layers):
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", k_caches[layer_idx])
            llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", v_caches[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", k_caches_out[layer_idx])
            llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", v_caches_out[layer_idx])

        llm_session.run_with_iobinding(llm_io_binding)
        llm_io_binding.synchronize_outputs()

        # Decide the next token using your preferred sampling strategy.
        argmax_sampling_io_binding.bind_ortvalue_input("logits", logits)
        argmax_sampling_session.run_with_iobinding(argmax_sampling_io_binding)
        argmax_sampling_io_binding.synchronize_outputs()
        output_tokens.append(next_token.numpy().item())

        # Stop if/when we get an ENDOFTEXT token before reaching maximum sequence length
        if output_tokens[-1] == tokenizer.eos_id:
            break

        # Update the embeddings for the next iteration
        update_embeddings_io_binding.bind_ortvalue_input("tokens", next_token)

        if idx == 0:
            llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
            update_embeddings_io_binding.bind_ortvalue_output("embeddings", x_increment)

        attn_mask_out, attn_mask = attn_mask, attn_mask_out
        cos_out, cos = cos, cos_out
        sin_out, sin = sin, sin_out
        k_caches, k_caches_out = k_caches_out, k_caches
        v_caches, v_caches_out = v_caches_out, v_caches

    after_time = time.perf_counter()
    print(f"Execution took {after_time - before_time:0.4f} seconds")

    output_str = tokenizer.decode(output_tokens)

    print(output_str)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="What is the lightest element?")
    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--disable_metacommands", action="store_true")
    args = parser.parse_args()
    run_llama_v2_io_binding(
        args.prompt,
        args.max_seq_len,
        args.max_gen_len,
        args.disable_metacommands,
    )
