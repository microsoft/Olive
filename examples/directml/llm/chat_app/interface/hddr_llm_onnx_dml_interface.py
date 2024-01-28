import gc
import os

import numpy as np
import onnxruntime
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state
from interface.base_interface import BaseLLMInterface
from transformers import AutoProcessor, AutoTokenizer


class LLMOnnxDmlInterface(BaseLLMInterface):
    def __init__(self, model_dir="", device="dml"):
        super().__init__()

        self.model_dir = model_dir
        self.device = device

    def initialize(self):
        # Create the ONNX sessions

        execution_provider = {
            "dml": "DmlExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }[self.device]

        providers = [execution_provider]

        self.max_seq_len = 2048

        llm_session_options = onnxruntime.SessionOptions()
        llm_session_options.add_free_dimension_override_by_name("batch_size", 1)
        llm_session_options.add_free_dimension_override_by_name("attention_mask_sequence_length", self.max_seq_len)
        llm_session_options.add_free_dimension_override_by_name("max_seq_len", self.max_seq_len)
        llm_session_options.add_free_dimension_override_by_name("seq_len_increment", 1)
        self.llm_session = onnxruntime.InferenceSession(
            os.path.join(self.model_dir, "decoder_model_merged.onnx"),
            sess_options=llm_session_options,
            providers=providers,
        )

        self.data_type = np.float16
        self.num_layers = 0
        for inputs_meta in self.llm_session._inputs_meta:
            if inputs_meta.name.startswith("cache.") and inputs_meta.name.endswith(".key"):
                self.num_layers += 1
                num_key_value_heads = inputs_meta.shape[1]
                head_dim = inputs_meta.shape[3]

        # Initialize the tokenizer and produce the initial tokens.
        if "llava" in self.model_dir:
            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        if self.tokenizer.chat_template is None:
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'user' %}"
                "USER: {{message['content']}} ASSISTANT:"
                "{% endif %}"
                "{% if message['role'] == 'assistant' %}"
                "{{message['content']}}"
                "{% endif %}"
                "{% endfor %}"
            )

        # Create the I/O bindings
        self.llm_io_binding = self.llm_session.io_binding()

        # Initialize the buffers
        self.tokens_increment = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, self.device)

        # Create the K and V caches.
        cache_shape = (1, num_key_value_heads, self.max_seq_len, head_dim)
        initial_cache = np.zeros(cache_shape, dtype=self.data_type)
        self.k_caches = []
        self.v_caches = []

        for _ in range(self.num_layers):
            self.k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.device))
            self.v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.device))

        if "llava" in self.model_dir:
            self.initial_prompt = [
                {"role": "user", "content": "I will ask you questions about the following image:\n<image>\n"},
                {"role": "assistant", "content": "OK!"},
            ]
        else:
            self.initial_prompt = [
                {"role": "user", "content": "Hey there I am a human that would like to have a conversation with you."},
                {"role": "assistant", "content": "Sure, I am happy to answer most questions."},
                {"role": "user", "content": "Great, I insist that we take turns."},
                {"role": "assistant", "content": "I agree, we should take turns."},
                {"role": "user", "content": "Great, can we also keep answers short?"},
                {"role": "assistant", "content": "Yes, short answers are usually best."},
            ]

    def shutdown(self):
        pass

    def generate_prompt_with_history(self, text, history, max_length=2048, image=None):
        prompt = []
        prompt.extend(self.initial_prompt)

        for dialogue in history:
            prompt.append({"role": "user", "content": dialogue[0]})
            prompt.append({"role": "assistant", "content": dialogue[1]})

        if self.processor is not None and image is not None:
            processed_inputs = self.processor(text=text, images=image, return_tensors="np")
            pixel_values = processed_inputs["pixel_values"].astype(np.float16)
            prompt.append({"role": "user", "content": text})
        else:
            prompt.append({"role": "user", "content": text})
            pixel_values = None

        tokens = self.tokenizer.apply_chat_template(prompt, return_tensors="np")

        if len(tokens) <= max_length:
            return tokens, pixel_values
        else:
            return None, None

    def greedy_search(
        self,
        input_ids,
        pixel_values,
        tokenizer,
        max_length: int,
        token_printing_step: int = 4,
    ):
        generated_tokens = []

        tokens = np.asarray(input_ids, dtype=np.int64)
        tokens = onnxruntime.OrtValue.ortvalue_from_numpy(tokens, self.device)

        seq_len = tokens.shape()[1]
        past_seq_len = 0

        if "llava" in self.model_dir:
            attention_mask = np.zeros((1, self.max_seq_len), dtype=np.int64)
            attention_mask[:, :seq_len] = 1
            self.llm_io_binding.bind_cpu_input("pixel_values", pixel_values)

        # Bind the main model's inputs/outputs
        self.llm_io_binding.bind_cpu_input("use_cache_branch", np.zeros([1], dtype=np.bool_))
        self.llm_io_binding.bind_output("logits", self.device)
        self.llm_io_binding.bind_ortvalue_input("tokens", tokens)
        self.llm_io_binding.bind_ortvalue_input("tokens_increment", self.tokens_increment)

        for i in range(max_length):
            if "llava" in self.model_dir:
                self.llm_io_binding.bind_cpu_input("attention_mask", attention_mask)
            else:
                if i == 0:
                    position_ids = np.arange(seq_len, dtype=np.int64).reshape((1, seq_len))
                    self.llm_io_binding.bind_cpu_input("position_ids", position_ids)
                else:
                    position_ids_increment = np.array(seq_len, dtype=np.int64).reshape((1, 1))
                    self.llm_io_binding.bind_cpu_input("position_ids_increment", position_ids_increment)

                seqlens_k = np.array(past_seq_len, dtype=np.int32, ndmin=1)
                self.llm_io_binding.bind_cpu_input("seqlens_k", seqlens_k)

            for layer_idx in range(self.num_layers):
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_input(f"cache.{layer_idx}.value", self.v_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.key", self.k_caches[layer_idx])
                self.llm_io_binding.bind_ortvalue_output(f"cache_out.{layer_idx}.value", self.v_caches[layer_idx])

            # Run the LLM
            self.llm_session.run_with_iobinding(self.llm_io_binding)

            # Decide the next token using your preferred sampling strategy.
            logits = self.llm_io_binding.get_outputs()[0].numpy()
            last_token_logits = logits[:, -1, :]
            next_token = np.argmax(last_token_logits, axis=-1, keepdims=True)
            generated_tokens.append(next_token.item())

            # Set the token for the next iteration
            self.llm_io_binding.bind_cpu_input("tokens_increment", next_token)

            if i % token_printing_step == 0:
                yield tokenizer.decode(generated_tokens, skip_special_tokens=True)

            if generated_tokens[-1] == tokenizer.eos_token_id:
                yield tokenizer.decode(generated_tokens, skip_special_tokens=True)
                return

            if i == 0:
                self.llm_io_binding.bind_cpu_input("use_cache_branch", np.ones([1], dtype=np.bool_))
                self.llm_io_binding.bind_output("logits", self.device)

                if "llava" in self.model_dir:
                    seq_len = logits.shape[1]
                    attention_mask = np.zeros((1, self.max_seq_len), dtype=np.int64)
                    attention_mask[:, :seq_len] = 1

            if "llava" in self.model_dir and seq_len < self.max_seq_len:
                attention_mask[:, seq_len] = 1

            past_seq_len = seq_len
            seq_len += 1

    def predict(
        self,
        text,
        chatbot,
        history,
        max_length_tokens,
        max_context_length_tokens,
        token_printing_step,
        image,
    ):
        if text == "":
            yield chatbot, history, "Empty context."
            return

        inputs, pixel_values = self.generate_prompt_with_history(
            text, history, max_length=max_context_length_tokens, image=image
        )

        if inputs is None:
            yield chatbot, history, "Input too long."
            return

        input_ids = inputs[:, -max_context_length_tokens:]

        x = input_ids

        for x in self.greedy_search(
            input_ids,
            pixel_values,
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
