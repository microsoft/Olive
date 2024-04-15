import gc
from pathlib import Path
from typing import List

import numpy as np
import onnxruntime
import torch
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state
from interface.base_interface import BaseLLMInterface
from sentencepiece import SentencePieceProcessor


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


class LlamaOnnxDmlInterface(BaseLLMInterface):
    def __init__(self, onnx_file="", sampling_onnx_file="", tokenizer_path=""):
        super().__init__()

        self.onnx_file = onnx_file
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

        llm_session_options = onnxruntime.SessionOptions()
        llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
        llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)

        self.llm_session = onnxruntime.InferenceSession(
            self.onnx_file,
            sess_options=llm_session_options,
            providers=providers,
        )

        sampling_session_options = onnxruntime.SessionOptions()
        sampling_session_options.add_free_dimension_override_by_name("batch_size", 1)
        self.sampling_session = onnxruntime.InferenceSession(
            self.sampling_onnx_file,
            sess_options=sampling_session_options,
            providers=providers,
        )

        self.data_type = np.float16

        self.hidden_size = 4096
        self.n_heads = 32
        self.n_layers = 32

        # Initialize the tokenizer and produce the initial tokens.
        self.tokenizer = Tokenizer(model_path=self.tokenizer_path)

        self.binding_device = "dml"

        # Create the K and V caches.
        self.head_dim = int(self.hidden_size / self.n_heads)

        # Create the I/O bindings
        self.sampling_io_binding = self.sampling_session.io_binding()
        self.llm_io_binding = self.llm_session.io_binding()

        logits_shape = (1, self.tokenizer.n_words)

        # Initialize the buffers
        self.logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            logits_shape, self.data_type, self.binding_device
        )
        self.tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, self.binding_device)

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

    def greedy_search(
        self,
        input_ids,
        tokenizer,
        max_length: int,
        token_printing_step: int = 4,
    ):
        generated_tokens = []

        tokens = np.asarray(input_ids, dtype=np.int64)
        tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, self.binding_device)

        seq_len = tokens.shape()[1]

        # Bind the main model's inputs/outputs
        self.llm_io_binding.bind_ortvalue_output("logits", self.logits)

        # Bind the sampling model's inputs/outputs
        self.sampling_io_binding.bind_ortvalue_output("next_token", self.tokens_increment)

        self.llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))

        padding = 512

        for i in range(max_length):
            # Setup the caches, mask and rotary embeddings
            if i == 0 or seq_len % padding == 0:
                padded_seq_len = padding * (seq_len // padding + 1)

                # Create the attention mask, which contains 1's for values that should stay intact, and 0's for values
                # that should get added to -10000
                attn_mask = np.pad(np.ones((1, seq_len)), ((0, 0), (padded_seq_len - seq_len, 0))).astype(np.int32)
                attn_mask = onnxruntime.OrtValue.ortvalue_from_numpy(attn_mask, self.binding_device)
                attn_mask_out = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
                    (1, padded_seq_len), np.int32, self.binding_device
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

            if i == 0:
                position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
                self.llm_io_binding.bind_cpu_input("position_ids", position_ids)
            else:
                position_ids_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))
                self.llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

            # Bind the inputs/outputs of the LLaMA model
            self.llm_io_binding.bind_ortvalue_input("tokens", tokens)
            self.llm_io_binding.bind_ortvalue_input("tokens_increment", self.tokens_increment)
            self.llm_io_binding.bind_ortvalue_input("attn_mask", attn_mask)
            self.llm_io_binding.bind_ortvalue_output("attn_mask_out", attn_mask_out)

            for layer_idx in range(self.n_layers):
                self.llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.value", self.v_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.key", self.k_caches_out[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.value", self.v_caches_out[layer_idx])

            # Run the LLaMA model
            self.llm_session.run_with_iobinding(self.llm_io_binding)
            self.llm_io_binding.synchronize_outputs()

            # Decide the next token using your preferred sampling strategy.
            self.sampling_io_binding.bind_ortvalue_input("logits", self.logits)
            self.sampling_session.run_with_iobinding(self.sampling_io_binding)
            self.sampling_io_binding.synchronize_outputs()
            generated_tokens.append(self.tokens_increment.numpy().item())

            if i % token_printing_step == 0:
                yield tokenizer.decode(generated_tokens)

            if generated_tokens[-1] == tokenizer.eos_id:
                yield tokenizer.decode(generated_tokens)
                return

            self.llm_io_binding.bind_ortvalue_input("tokens_increment", self.tokens_increment)

            if i == 0:
                self.llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))

            attn_mask_out, attn_mask = attn_mask, attn_mask_out
            self.k_caches, self.k_caches_out = self.k_caches_out, self.k_caches
            self.v_caches, self.v_caches_out = self.v_caches_out, self.v_caches
            seq_len += 1

    def predict(
        self,
        text,
        chatbot,
        history,
        max_length_tokens,
        max_context_length_tokens,
        token_printing_step,
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return

        inputs = self.generate_prompt_with_history(text, history, self.tokenizer, max_length=max_context_length_tokens)

        if inputs is None:
            yield chatbot, history, "Input too long."
            return
        else:
            _, inputs = inputs

        input_ids = inputs[:, -max_context_length_tokens:]

        # global total_count
        self.total_count += 1
        print(self.total_count)

        for x in self.greedy_search(
            input_ids,
            self.tokenizer,
            max_length=max_length_tokens,
            token_printing_step=token_printing_step,
        ):
            sentence = x

            if is_stop_word_or_prefix(sentence, ["[|Human|]", "[|AI|]"]) is False:
                if "[|Human|]" in sentence:
                    sentence = sentence[: sentence.index("[|Human|]")].strip()
                if "[|AI|]" in sentence:
                    sentence = sentence[: sentence.index("[|AI|]")].strip()
                sentence = sentence.strip()
                a, b = [[y[0], convert_to_markdown(y[1])] for y in history] + [[text, convert_to_markdown(sentence)]], [
                    *history,
                    [text, sentence],
                ]
                yield a, b, "Generating..."

            if shared_state.interrupted:
                shared_state.recover()
                try:
                    yield a, b, "Stop: Success"
                    return
                except Exception as e:
                    print(type(e).__name__, e)

        del input_ids
        gc.collect()

        try:
            yield a, b, "Generate: Success"
        except Exception as e:
            print(type(e).__name__, e)

        return

    def retry(self, chatbot, history, max_length_tokens, max_context_length_tokens, token_printing_step):
        if len(history) == 0:
            yield chatbot, history, "Empty context"
            return
        chatbot.pop()
        inputs = history.pop()[0]
        yield from self.predict(
            inputs,
            chatbot,
            history,
            max_length_tokens,
            max_context_length_tokens,
            token_printing_step,
        )
