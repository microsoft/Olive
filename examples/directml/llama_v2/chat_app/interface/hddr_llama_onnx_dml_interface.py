import gc
import os
from typing import List

import numpy as np
import onnxruntime
import torch
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state
from interface.base_interface import BaseLLMInterface
from sentencepiece import SentencePieceProcessor


def rotary_mat(
    hidden_size: int,
    n_heads: int,
    max_seq_len: int,
    theta: float = 10000.0,
    head_scale=1.0,
    dtype=np.float16,
) -> tuple[np.ndarray, np.ndarray]:
    head_dim = head_scale * hidden_size / n_heads

    pos = np.arange(0, 2 * (head_dim // 2), step=2, dtype=dtype)
    freqs = 1.0 / (theta ** (pos / head_dim))

    idx = np.arange(max_seq_len, dtype=dtype)
    freqs = np.outer(idx, freqs)

    cos = np.reshape(np.cos(freqs), [1, max_seq_len, 1, -1])
    sin = np.reshape(np.sin(freqs), [1, max_seq_len, 1, -1])

    return cos, sin


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


class LlamaOnnxDmlInterface(BaseLLMInterface):
    def __init__(self, onnx_file="", update_embeddings_onnx_file="", sampling_onnx_file="", tokenizer_path=""):
        super().__init__()

        self.onnx_file = onnx_file
        self.update_embeddings_onnx_file = update_embeddings_onnx_file
        self.sampling_onnx_file = sampling_onnx_file
        self.tokenizer_path = tokenizer_path

        self.total_count = 0

    def initialize(self):
        # Create the ONNX sessions

        providers = [
            (
                "DmlExecutionProvider",
                {
                    "disable_metacommands": True,
                    "enable_dynamic_graph_fusion": True,
                },
            )
        ]

        self.llm_session = onnxruntime.InferenceSession(
            self.onnx_file,
            sess_options=onnxruntime.SessionOptions(),
            providers=providers,
        )

        self.update_embeddings_session = onnxruntime.InferenceSession(
            self.update_embeddings_onnx_file,
            sess_options=onnxruntime.SessionOptions(),
            providers=providers,
        )

        self.sampling_session = onnxruntime.InferenceSession(
            self.sampling_onnx_file,
            sess_options=onnxruntime.SessionOptions(),
            providers=providers,
        )

        # get the data type used by the model
        data_type_str = self.llm_session.get_inputs()[0].type
        if data_type_str == "tensor(float16)":
            self.data_type = np.float16
        elif data_type_str == "tensor(float32)":
            self.data_type = np.float32
        else:
            raise Exception(f"Unknown data type {data_type_str}")

        # Get the relevant shapes so we can create the inputs
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name == "x":
                x_shape = inputs_meta.shape
            elif inputs_meta.name == "cache.0.key":
                self.n_heads = inputs_meta.shape[1]

        self.hidden_size = x_shape[2]
        self.n_layers = 32

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)

        self.binding_device = "dml"

        # Create the K and V caches.
        self.head_dim = int(self.hidden_size / self.n_heads)

        # Create the I/O bindings
        self.sampling_io_binding = self.sampling_session.io_binding()
        self.update_embeddings_io_binding = self.update_embeddings_session.io_binding()
        self.llm_io_binding = self.llm_session.io_binding()

        logits_shape = (1, self.tokenizer.n_words)

        # Initialize the buffers
        self.logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            logits_shape, self.data_type, self.binding_device
        )
        self.next_token = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1,), np.int64, self.binding_device)
        self.x_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            (1, 1, self.hidden_size), self.data_type, self.binding_device
        )

        self.k_caches = [None] * self.n_layers
        self.v_caches = [None] * self.n_layers
        self.k_caches_out = [None] * self.n_layers
        self.v_caches_out = [None] * self.n_layers

    def shutdown(self):
        pass

    def generate_prompt_with_history(self, text, history, tokenizer, max_length=2048):
        prompt = "[|Human|]Hey there I am a human that would like to have\
a conversation with you.\n[|AI|]Sure, I am happy to answer most questions\
\n[|Human|]Great, I insist that we take turns.\n[|AI|]I agree, we should\
 take turns.\n[|Human|]Great, can we also keep answers short\n[|AI|]Yes, \
short answers are usually best"

        history = ["\n[|Human|]{}\n[|AI|]{}".format(x[0], x[1]) for x in history]
        history.append("\n[|Human|]{}\n[|AI|]".format(text))
        history_text = ""
        flag = False
        for x in history[::-1]:
            # tokens = self.tokenizer.encode(text, bos=True, eos=False)
            if len(self.tokenizer.encode(prompt + history_text + x, bos=True, eos=False)) <= max_length:
                history_text = x + history_text
                flag = True
            else:
                break
        if flag:
            return prompt + history_text, torch.tensor(
                self.tokenizer.encode(prompt + history_text, bos=True, eos=False)
            ).unsqueeze(0)
        else:
            return None

    def sample_logits(
        self,
        logits: np.ndarray,
        sampling_method: str = "greedy",
        sampling_value: float = None,
        temperature: float = 1.0,
    ) -> np.ndarray:
        if temperature == 0 or sampling_method == "greedy":
            next_token = np.argmax(logits, axis=-1).astype(np.int64)

        elif sampling_method == "top_k" or sampling_method == "top_p":
            assert sampling_value is not None

            # temperature, converting to probabilities and sorting are common to both top-k and top-p
            # convert logits to 32-bit float to avoid numerical issues with np.exp
            logits = logits.astype(np.float32)
            # Scale the logits by the temperature
            logits /= temperature
            # Convert logits to probabilities
            probs = np.exp(logits) / np.sum(np.exp(logits))
            # Sort th probabilities and indexes
            sorted_probs = np.sort(probs)[:, ::-1]
            sorted_indices = np.argsort(probs)[:, ::-1]

            # find the index of interest for each of the methods.
            if sampling_method == "top_k":
                index_of_interest = int(sampling_value)
            elif sampling_method == "top_p":
                p = sampling_value
                cumulative_probs = np.cumsum(sorted_probs, axis=-1)
                # find the value of the first cumalitive probability that exceeds p
                for index_of_interest, cumulative_prob in enumerate(cumulative_probs[0]):
                    if cumulative_prob > p:
                        break

            probs_of_interest = sorted_probs[:, : index_of_interest + 1]
            indices_of_interest = sorted_indices[:, : index_of_interest + 1]
            # Normalize the probabilities and select the next token
            probs_of_interest /= np.sum(probs_of_interest)
            next_token = np.array([np.random.choice(indices_of_interest[0], p=probs_of_interest[0])])
        else:
            raise Exception(f"Unknown sampling method {sampling_method}")

        return next_token

    def greedy_search(
        self,
        input_ids,
        model,
        tokenizer,
        stop_words: list,
        max_length: int,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = 25,
        token_printing_step: int = 4,
    ):
        generated_tokens = []

        tokens = onnxruntime.OrtValue.ortvalue_from_numpy(
            np.asarray(np.squeeze(input_ids), dtype=np.int64), self.binding_device
        )
        x = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            (1, tokens.shape()[0], self.hidden_size), self.data_type, self.binding_device
        )

        seq_len = tokens.shape()[0]

        # Bind the main model's inputs/outputs
        self.llm_io_binding.bind_ortvalue_output("logits", self.logits)

        # Bind the sampling model's inputs/outputs
        self.sampling_io_binding.bind_ortvalue_output("next_token", self.next_token)

        # Bind the embeddings updating model's inputs/outputs
        self.update_embeddings_io_binding.bind_ortvalue_input("tokens", tokens)
        self.update_embeddings_io_binding.bind_ortvalue_output("embeddings", x)

        self.llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))

        padding = 512

        for i in range(max_length):
            # Update the embeddings
            self.update_embeddings_session.run_with_iobinding(self.update_embeddings_io_binding)
            self.update_embeddings_io_binding.synchronize_outputs()

            # Setup the caches, mask and rotary embeddings
            if i == 0 or seq_len % padding == 0:
                padded_seq_len = padding * (seq_len // padding + 1)
                cos, sin = rotary_mat(self.hidden_size, self.n_heads, padded_seq_len, head_scale=1.0)

                if i > 0:
                    cos = np.roll(cos, padding, axis=1)
                    sin = np.roll(sin, padding, axis=1)

                cos = onnxruntime.OrtValue.ortvalue_from_numpy(cos, self.binding_device)
                cos_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    (1, padded_seq_len, 1, 64), self.data_type, self.binding_device
                )

                sin = onnxruntime.OrtValue.ortvalue_from_numpy(sin, self.binding_device)
                sin_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    (1, padded_seq_len, 1, 64), self.data_type, self.binding_device
                )

                # Create the attention mask, which contains 1's for values that should stay intact, and 0's for values
                # that should get added to -10000
                attn_mask = np.tril(np.ones((1, padded_seq_len, padded_seq_len))).astype(np.int32)

                if i > 0:
                    attn_mask[:, -1, :padding] = 0

                attn_mask = onnxruntime.OrtValue.ortvalue_from_numpy(attn_mask, self.binding_device)
                attn_mask_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    (1, padded_seq_len, padded_seq_len), np.int32, self.binding_device
                )

                for layer_idx in range(self.n_layers):
                    if i == 0:
                        self.k_caches[layer_idx] = np.zeros(
                            (1, self.n_heads, padded_seq_len, self.head_dim), dtype=self.data_type
                        )
                        self.v_caches[layer_idx] = np.zeros(
                            (1, self.n_heads, padded_seq_len, self.head_dim), dtype=self.data_type
                        )
                    else:
                        self.k_caches[layer_idx] = np.pad(
                            self.k_caches[layer_idx].numpy(), ((0, 0), (0, 0), (padding, 0), (0, 0))
                        )
                        self.v_caches[layer_idx] = np.pad(
                            self.v_caches[layer_idx].numpy(), ((0, 0), (0, 0), (padding, 0), (0, 0))
                        )

                    self.k_caches[layer_idx] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        self.k_caches[layer_idx], self.binding_device
                    )
                    self.v_caches[layer_idx] = onnxruntime.OrtValue.ortvalue_from_numpy(
                        self.v_caches[layer_idx], self.binding_device
                    )
                    self.k_caches_out[layer_idx] = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                        self.k_caches[layer_idx].shape(), self.data_type, self.binding_device
                    )
                    self.v_caches_out[layer_idx] = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                        self.v_caches[layer_idx].shape(), self.data_type, self.binding_device
                    )

            # Bind the inputs/outputs of the LLaMA model
            self.llm_io_binding.bind_ortvalue_input("x", x)
            self.llm_io_binding.bind_ortvalue_input("x_increment", self.x_increment)
            self.llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
            self.llm_io_binding.bind_ortvalue_input("cos", cos)
            self.llm_io_binding.bind_ortvalue_input("sin", sin)
            self.llm_io_binding.bind_ortvalue_output("attn_mask_out", attn_mask_out)
            self.llm_io_binding.bind_ortvalue_output("cos_out", cos_out)
            self.llm_io_binding.bind_ortvalue_output("sin_out", sin_out)

            for layer_idx in range(self.n_layers):
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", self.v_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", self.k_caches_out[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", self.v_caches_out[layer_idx])

            # Run the LLaMA model
            self.llm_session.run_with_iobinding(self.llm_io_binding)
            self.llm_io_binding.synchronize_outputs()

            # Decide the next token using your preferred sampling strategy.
            self.sampling_io_binding.bind_ortvalue_input("logits", self.logits)
            self.sampling_session.run_with_iobinding(self.sampling_io_binding)
            self.sampling_io_binding.synchronize_outputs()
            generated_tokens.append(self.next_token.numpy().item())

            if i % token_printing_step == 0:
                yield tokenizer.decode(generated_tokens)

            if generated_tokens[-1] == tokenizer.eos_id:
                yield tokenizer.decode(generated_tokens)
                return

            # Update the embeddings for the next iteration
            self.update_embeddings_io_binding.bind_ortvalue_input("tokens", self.next_token)

            if i == 0:
                self.llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
                self.update_embeddings_io_binding.bind_ortvalue_output("embeddings", self.x_increment)

            attn_mask_out, attn_mask = attn_mask, attn_mask_out
            cos_out, cos = cos, cos_out
            sin_out, sin = sin, sin_out
            self.k_caches, self.k_caches_out = self.k_caches_out, self.k_caches
            self.v_caches, self.v_caches_out = self.v_caches_out, self.v_caches
            seq_len += 1

    def predict(
        self,
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
        token_printing_step,
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return
        try:
            self.llm_session
        except (ValueError, RuntimeError, TypeError):
            yield [[text, "No Model Found"]], [], "No Model Found"
            return

        inputs = self.generate_prompt_with_history(text, history, self.tokenizer, max_length=max_context_length_tokens)

        if inputs is None:
            yield chatbot, history, "Input too long."
            return
        else:
            prompt, inputs = inputs

        input_ids = inputs[:, -max_context_length_tokens:]

        # global total_count
        self.total_count += 1
        print(self.total_count)

        x = input_ids

        for x in self.greedy_search(
            input_ids,
            self.llm_session,
            self.tokenizer,
            stop_words=["[|Human|]", "[|AI|]"],
            max_length=max_length_tokens,
            temperature=temperature,
            top_p=top_p,
            token_printing_step=token_printing_step,
        ):
            if is_stop_word_or_prefix(x, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in x:
                    x = x[: x.index("[|Human|]")].strip()
                if "[|AI|]" in x:
                    x = x[: x.index("[|AI|]")].strip()
                x = x.strip()
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [
                    [text, convert_to_markdown(x)]
                ], history + [[text, x]]
                yield a, b, "Generating..."

            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except Exception as e:
                    print(type(e).__name__, e)
                    pass

        del input_ids
        gc.collect()

        try:
            yield a, b, "Generate: Success"
        except Exception as e:
            print(type(e).__name__, e)
            pass

        return

    def retry(
        self,
        text,
        chatbot,
        history,
        top_p,
        temperature,
        max_length_tokens,
        max_context_length_tokens,
    ):
        if len(history) == 0:
            yield chatbot, history, "Empty context"
            return
        chatbot.pop()
        inputs = history.pop()[0]
        for x in self.predict(
            inputs,
            chatbot,
            history,
            top_p,
            temperature,
            max_length_tokens,
            max_context_length_tokens,
        ):
            yield x
