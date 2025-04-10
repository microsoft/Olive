import gc
import os
import sys
from pathlib import Path

import numpy as np
import onnxruntime
from app_modules.utils import convert_to_markdown, is_stop_word_or_prefix, shared_state
from interface.base_interface import BaseLLMInterface
from transformers import AutoProcessor, AutoTokenizer

current_dir = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.join(current_dir, "..", "..", ".."))


class LLMOnnxDmlInterface(BaseLLMInterface):
    def __init__(self, model_dir="", device="dml"):
        super().__init__()

        self.model_dir = model_dir
        self.device = device

    def initialize(self):
        execution_provider = {
            "dml": "DmlExecutionProvider",
            "cuda": "CUDAExecutionProvider",
        }[self.device]

        # Initialize the providers
        providers = [
            (
                execution_provider,
                {
                    "device_id": 0,
                },
            )
        ]

        if self.device == "cuda":
            providers[0][1]["enable_cuda_graph"] = True

        self.max_seq_len = 2048

        llm_session_options = onnxruntime.SessionOptions()
        llm_session_options.add_session_config_entry("ep.dml.enable_graph_capture", "1")

        self.llm_session = onnxruntime.InferenceSession(
            os.path.join(self.model_dir, "model.onnx"),
            sess_options=llm_session_options,
            providers=providers,
        )

        self.data_type = np.float16
        self.num_layers = 0
        for inputs_meta in self.llm_session._inputs_meta:  # pylint: disable=protected-access
            if inputs_meta.name.startswith("past_key_values.") and inputs_meta.name.endswith(".key"):
                self.num_layers += 1
                num_key_value_heads = inputs_meta.shape[1]
                head_dim = inputs_meta.shape[3]

        for outputs_meta in self.llm_session._outputs_meta:
            if outputs_meta.name == "logits":
                vocab_size = outputs_meta.shape[2]

        # Initialize the tokenizer and produce the initial tokens.
        if "llava" in self.model_dir:
            self.processor = AutoProcessor.from_pretrained(self.model_dir)
            self.tokenizer = self.processor.tokenizer
        else:
            self.processor = None
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)

        from llm.chat_templates import get_chat_template
        from llm.model_type_mapping import get_model_type

        model_name = Path(self.model_dir).stem
        model_type = get_model_type(model_name)

        self.tokenizer.chat_template = get_chat_template(model_type) or self.tokenizer.chat_template

        # Create the I/O bindings
        self.llm_io_binding = self.llm_session.io_binding()

        # Initialize the buffers on the GPU
        cache_shape = (1, num_key_value_heads, self.max_seq_len, head_dim)
        initial_cache = np.zeros(cache_shape, dtype=self.data_type)
        self.k_caches = []
        self.v_caches = []

        for _ in range(self.num_layers):
            self.k_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.device))
            self.v_caches.append(onnxruntime.OrtValue.ortvalue_from_numpy(initial_cache, self.device))

        self.position_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, self.device)
        self.attention_mask_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            (1, self.max_seq_len), np.int64, self.device
        )
        self.input_ids_ortvalue = onnxruntime.OrtValue.ortvalue_from_shape_and_type((1, 1), np.int64, self.device)
        self.increment_logits = onnxruntime.OrtValue.ortvalue_from_shape_and_type(
            (1, 1, vocab_size), self.data_type, self.device
        )

        # Bind the inputs and outputs
        for layer_idx in range(self.num_layers):
            self.llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.key", self.k_caches[layer_idx])
            self.llm_io_binding.bind_ortvalue_input(f"past_key_values.{layer_idx}.value", self.v_caches[layer_idx])
            self.llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.key", self.k_caches[layer_idx])
            self.llm_io_binding.bind_ortvalue_output(f"present.{layer_idx}.value", self.v_caches[layer_idx])

        self.llm_io_binding.bind_ortvalue_input("attention_mask", self.attention_mask_ortvalue)

        # Set the initial prompt
        if "llava" in self.model_dir:
            self.initial_prompt = [
                {"role": "user", "content": "I will ask you questions about the following image:\n<image>\n"},
                {"role": "assistant", "content": "OK!"},
            ]
        else:
            self.initial_prompt = []

    def shutdown(self):
        pass

    def generate_prompt_with_history(self, text, history, max_length=2048, image=None):
        prompt = []
        prompt.extend(self.initial_prompt)

        for dialog in history:
            prompt.append({"role": "user", "content": dialog[0]})
            prompt.append({"role": "assistant", "content": dialog[1]})

        if self.processor is not None and image is not None:
            processed_inputs = self.processor(text=text, images=image, return_tensors="np")
            pixel_values = processed_inputs["pixel_values"].astype(np.float16)
            prompt.append({"role": "user", "content": text})
        else:
            prompt.append({"role": "user", "content": text})
            pixel_values = None

        input_ids = self.tokenizer.apply_chat_template(prompt, return_tensors="np")

        if len(input_ids) <= max_length:
            return input_ids, pixel_values
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
        output_tokens = []

        initial_input_ids = np.asarray(input_ids, dtype=np.int64)
        initial_input_ids = onnxruntime.OrtValue.ortvalue_from_numpy(initial_input_ids, self.device)

        sequence_length = initial_input_ids.shape()[1]

        if "llava" in self.model_dir:
            attention_mask = np.zeros((1, self.max_seq_len), dtype=np.int64)
            attention_mask[:, :sequence_length] = 1
            self.llm_io_binding.bind_cpu_input("pixel_values", pixel_values)

        self.llm_io_binding.bind_output("logits", self.device)
        self.llm_io_binding.bind_ortvalue_input("input_ids", initial_input_ids)

        initial_position_ids = np.arange(sequence_length, dtype=np.int64).reshape((1, sequence_length))
        self.llm_io_binding.bind_cpu_input("position_ids", initial_position_ids)

        attention_mask = np.pad(
            np.ones((1, sequence_length), dtype=np.int64), ((0, 0), (0, self.max_seq_len - sequence_length))
        )

        run_options = onnxruntime.RunOptions()

        for idx in range(max_length):
            if idx > 0:
                position_ids = np.array(sequence_length - 1, dtype=np.int64, ndmin=2)
                self.position_ids_ortvalue.update_inplace(position_ids)

            if idx == 1:
                self.llm_io_binding.bind_ortvalue_input("position_ids", self.position_ids_ortvalue)
                self.llm_io_binding.bind_ortvalue_input("input_ids", self.input_ids_ortvalue)

            attention_mask[0, (sequence_length - 1) % self.max_seq_len] = 1
            self.attention_mask_ortvalue.update_inplace(attention_mask)

            # Run the LLM
            if idx == 0:
                run_options.add_run_config_entry("gpu_graph_id", "-1")
            elif idx == 1:
                run_options.add_run_config_entry("gpu_graph_id", "1")

            self.llm_session.run_with_iobinding(self.llm_io_binding, run_options)

            # Decide the next token using your preferred sampling strategy.
            logits = self.llm_io_binding.get_outputs()[-1].numpy()[:, -1, :]
            next_token = np.argmax(logits, axis=-1, keepdims=True)
            output_tokens.append(next_token.item())

            # Set the token for the next iteration
            self.input_ids_ortvalue.update_inplace(next_token)

            if idx % token_printing_step == 0:
                yield tokenizer.decode(output_tokens, skip_special_tokens=True)

            if output_tokens[-1] == tokenizer.eos_token_id:
                yield tokenizer.decode(output_tokens, skip_special_tokens=True)
                return

            if idx == 0:
                self.llm_io_binding.bind_ortvalue_output("logits", self.increment_logits)

                if "llava" in self.model_dir:
                    sequence_length = logits.shape[1]
                    attention_mask[:, : (sequence_length % self.max_seq_len)] = 1

            sequence_length += 1

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

        human_tokens = [
            "[|Human|]",
            "Human:",
            "### HUMAN:",
            "### User:",
            "USER:",
            "<|im_start|>user",
            "<|user|>",
            "### Instruction:",
            "GPT4 Correct User:",
        ]

        ai_tokens = [
            "[|AI|]",
            "AI:",
            "### RESPONSE:",
            "### Response:",
            "ASSISTANT:",
            "<|im_start|>assistant",
            "<|assistant|>",
            "GPT4 Correct Assistant:",
            "### Assistant:",
        ]

        for x in self.greedy_search(
            input_ids,
            pixel_values,
            self.tokenizer,
            max_length=max_length_tokens,
            token_printing_step=token_printing_step,
        ):
            sentence = x

            if is_stop_word_or_prefix(sentence, ["[|Human|]", "[|AI|]", "Human:", "AIL"]) is False:
                for human_token in human_tokens:
                    if human_token in sentence:
                        sentence = sentence[: sentence.index(human_token)].strip()
                        break

                for ai_token in ai_tokens:
                    if ai_token in sentence:
                        sentence = sentence[: sentence.index(ai_token)].strip()
                        break
                sentence = sentence.strip()
                a, b = (
                    [[y[0], convert_to_markdown(y[1])] for y in history] + [[text, convert_to_markdown(sentence)]],
                    [
                        *history,
                        [text, sentence],
                    ],
                )
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
