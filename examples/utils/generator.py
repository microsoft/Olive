# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from kv_cache_utils import DynamicCache, DynamicIOBoundCache, GQASharedCache, StaticCache, StaticIOBoundCache
from onnxruntime import OrtValue, set_default_logger_severity
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from olive.common.hf.model_io import replace_with_extended_mask
from olive.common.ort_inference import get_ort_inference_session
from olive.common.utils import load_weights
from olive.model.handler.mixin.onnx_graph import OnnxGraphMixin

if TYPE_CHECKING:
    from kv_cache_utils import Cache, IOBoundCache
    from numpy.typing import NDArray
    from onnxruntime import InferenceSession

# ort logs warnings about default initializers
set_default_logger_severity(3)


class ORTGenerator:
    def __init__(
        self,
        model_path: Union[str, Tuple[str, str]],
        tokenizer: PreTrainedTokenizer,
        execution_provider: str,
        device_id: int = 0,
        adapters: Dict[str, Any] = None,
        has_gqa: Optional[bool] = None,
    ):
        """Initialize the generator.

        :param model_path: Path to the model.
        :param tokenizer: The tokenizer to use.
        :param execution_provider: The execution provider to use.
        :param device_id: The device id to use.
        :param adapters: Dictionary of adapter information. Each key is the adapter name and the value is a
            dictionary with the following keys
            - "weights": Path to the weights file or dictionary of weights.
            - "template": The template to use for the adapter.
            If not provided, will assume the model has no adapters.
        """
        if isinstance(model_path, str):
            self.model_path = (model_path,)
        else:
            assert len(model_path) == 2, "model_path should be a string or a tuple of two strings."
            self.model_path = model_path
        self.tokenizer = tokenizer
        self.execution_provider = execution_provider
        self.device_id = device_id
        self.adapter_info = adapters or {"default": {"weights": None, "template": None}}

        # Determine attention type
        if has_gqa is not None:
            self.has_gqa = has_gqa
        else:
            for node in onnx.load(self.model_path[-1], load_external_data=False).graph.node:
                if node.op_type == "GroupQueryAttention":
                    self.has_gqa = True
                    break
            else:
                self.has_gqa = False

        # Get io info
        # use the generator graph for io info since it has all the inputs and outputs
        self.input_info, self.output_info = ORTGenerator.get_io_info(self.model_path[-1])
        self.cache_info = {"past_names": [], "present_names": [], "dtype": None, "num_kv_heads": None, "head_dim": None}
        # static shapes
        self.batch_size = None
        self.prompt_len = None
        self.cache_len = None
        for i_name, i_info in self.input_info.items():
            if ".key" not in i_name and ".value" not in i_name:
                continue
            if not self.cache_info["past_names"]:
                self.cache_info["dtype"] = i_info["dtype"]
                self.cache_info["num_kv_heads"] = i_info["shape"][1]
                self.cache_info["head_dim"] = i_info["shape"][3]
                if isinstance(i_info["shape"][2], int):
                    self.cache_len = i_info["shape"][2]
            self.cache_info["past_names"].append(i_name)
        for o_name in self.output_info:
            if ".key" in o_name or ".value" in o_name:
                self.cache_info["present_names"].append(o_name)
        if len(self.model_path) == 2:
            c_i, c_o = ORTGenerator.get_io_info(self.model_path[0])
            self.batch_size, self.prompt_len = c_i["input_ids"]["shape"]
            if self.cache_len is not None:
                assert (
                    c_o[self.cache_info["present_names"][0]]["shape"][2] == self.cache_len
                ), "prefill present length should match decoder past length."
                assert self.cache_len == self.prompt_len, "Static cache length must match prompt length."
                past_name = self.cache_info["past_names"][0]
                assert past_name not in c_i or c_i[past_name]["shape"][2] == 0, "prefill past should be empty."

        # Determine device to use for io binding
        # NOTE: QNNExecutionProvider does not support IO binding but keep it for future compatibility
        ep_to_device_map = {
            "CPUExecutionProvider": "cpu",
            "CUDAExecutionProvider": "cuda",
            "DmlExecutionProvider": "dml",
            "QNNExecutionProvider": "cpu",
        }
        self.device = ep_to_device_map[self.execution_provider]

        # create the session
        self.sessions, self.adapters = self.prepare_sessions(
            self.model_path, self.execution_provider, self.device, self.device_id, self.adapter_info
        )
        self.default_adapter = next(iter(self.adapters.keys()))

    @staticmethod
    def get_io_info(model_path: str) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        io_config = OnnxGraphMixin.get_graph_io_config(model_path)
        return tuple(
            {
                name: {"dtype": dtype, "shape": shape}
                for name, dtype, shape in zip(
                    io_config[f"{prefix}_names"], io_config[f"{prefix}_types"], io_config[f"{prefix}_shapes"]
                )
            }
            for prefix in ["input", "output"]
        )

    @property
    def use_position_ids(self) -> bool:
        """Whether the model has position ids."""
        return "position_ids" in self.input_info

    @property
    def extended_attention_mask(self) -> bool:
        """Whether the model has extended attention mask."""
        return len(self.input_info["attention_mask"]["shape"]) == 4

    @staticmethod
    def prepare_sessions(
        model_path: str, execution_provider: str, device: str, device_id: int, adapter_info: Dict[str, Any]
    ) -> Tuple[Dict[str, "InferenceSession"], Dict[str, Dict[str, Any]]]:
        """Prepare the sessions and adapters for the model.

        :param model_path: Path to the model.
        :param execution_provider: The execution provider to use.
        :param device: The device to use.
        :param device_id: The device id to use.
        :param adapter_info: Dictionary of adapter information. Each key is the adapter name and the value is a
            dictionary with the following keys
            - "weights": Path to the weights file or dictionary of weights.
            - "template": The template to use for the adapter.
        :return: Tuple of sessions and adapters.
            Sessions is a dictionary of session names to InferenceSession objects.
            Adapters is a dictionary of adapter names to dictionaries with the following keys
            - "template": The template to use for the adapter.
            - "numpy_inputs": The numpy inputs for the adapter.
            - "ortvalue_inputs": The OrtValue inputs for the adapter.
        """
        sessions = {}
        adapters = {}
        inference_settings = {"execution_provider": [execution_provider]}
        # TODO(jambayk): test and enable graph for cuda and dml
        sessions["prefill"] = get_ort_inference_session(model_path[0], inference_settings, device_id=device_id)
        if len(model_path) == 2:
            sessions["iterator"] = get_ort_inference_session(model_path[1], inference_settings, device_id=device_id)
        else:
            sessions["iterator"] = sessions["prefill"]

        for name, info in adapter_info.items():
            adapters[name] = {
                "template": info.get("template"),
            }
            if info.get("weights") is not None:
                np_weights = load_weights(info["weights"]) if isinstance(info["weights"], str) else info["weights"]
                # TODO(jambayk): provide an option to lazy load the ortvalues if needed
                # for example, there is not enough memory to load all adapters at once
                if execution_provider != "QNNExecutionProvider":
                    ort_values = {k: OrtValue.ortvalue_from_numpy(v, device, device_id) for k, v in np_weights.items()}
                else:
                    ort_values = None
                adapters[name].update({"numpy_inputs": np_weights, "ortvalue_inputs": ort_values})
        return sessions, adapters

    def get_adapter(self, name: Optional[str] = None, use_io_binding: bool = False) -> Tuple[str, Dict[str, Any]]:
        """Get the session, template and inputs for the specified adapter.

        :param name: Name of the adapter to use. If None, the default adapter is used.
        :param use_io_binding: Whether to use IO binding for the adapter.
        :return: Tuple of template and inputs.
            The adapter_inputs are either numpy arrays or OrtValues depending on use_io_binding.
        """
        if name is None:
            name = self.default_adapter
        elif name not in self.adapters:
            raise ValueError(f"Adapter {name} not found.")

        template = self.adapters[name]["template"]
        inputs = (
            self.adapters[name].get("ortvalue_inputs") if use_io_binding else self.adapters[name].get("numpy_inputs")
        )
        return template, inputs

    def generate(
        self,
        prompt: Union[Union[str, Tuple[str]], List[Union[str, Tuple[str]]]],
        adapter: Optional[str] = None,
        max_gen_len: int = 128,
        use_io_binding: bool = False,
        cache_type: str = "auto",
        max_cache_len: int = 1024,
        cache_backend: str = "ort",
    ) -> Union[str, List[str]]:
        """Generate text from the model given a prompt.

        :param prompt: The prompt to generate text from. Can be a string/string-tuples or a list.
        :param adapter: The adapter to use for generation. If None, the default adapter is used.
        :param max_gen_len: The maximum length of the generated text.
        :param use_io_binding: Whether to use IO binding for the generation.
        :param cache_type: The type of cache to use for the generation. Can be "auto", "static" or "dynamic".
        :param max_cache_len: The maximum length of the cache for static cache.
        :param cache_backend: The backend to use for static cache. Can be "ort" or "torch".
        :return: The generated text.
        """
        if cache_type == "auto":
            cache_type = "static" if self.cache_len is not None else "dynamic"

        # template and adapter inputs
        # the adapter input is dictionary of adapter weights
        template, adapter_inputs = self.get_adapter(adapter, use_io_binding)

        # get the inputs for prompt processing
        inputs, cache = self.get_initial_inputs(
            prompt, template, use_io_binding, cache_type, max_cache_len, cache_backend
        )
        inputs = {**inputs, **(adapter_inputs or {})}

        generated_tokens = inputs["input_ids"].numpy() if use_io_binding else inputs["input_ids"].copy()
        batch_size, prompt_length = generated_tokens.shape
        has_eos = np.zeros(batch_size, dtype=bool)

        # buffers to keep numpy copy of model inputs, don't want to keep going back and forth between OrtValue and numpy
        np_buffers = {
            "attention_mask": (
                inputs.pop("attention_mask_2d").numpy() if use_io_binding else inputs.pop("attention_mask_2d")
            )
        }
        valid_prompt_len = int(np_buffers["attention_mask"].sum(axis=-1).max())
        if self.use_position_ids:
            np_buffers["position_ids"] = (
                np_buffers["attention_mask"]
                .sum(axis=1)
                .reshape(batch_size, 1)
                .astype(self.input_info["position_ids"]["dtype"])
            )

        session = None
        io_binding = None
        for idx in tqdm(range(max_gen_len)):
            if idx == 0:
                session = self.sessions["prefill"]
                io_binding = session.io_binding() if use_io_binding else None
            elif idx == 1:
                session = self.sessions["iterator"]
                io_binding = session.io_binding() if use_io_binding else None

            # print(np_buffers["attention_mask"])

            if use_io_binding:
                if idx < 2:
                    # need to bind logits twice, once for prompt processing and once for token generation
                    # shapes: (batch_size, prompt_length, vocab_size) and (batch_size, 1, vocab_size)
                    io_binding.bind_output("logits", self.device, self.device_id)
                for k, v in inputs.items():
                    io_binding.bind_ortvalue_input(k, v)
                cache.bind_kv_io(io_binding)

                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

                outputs = io_binding.get_outputs()
                logits = outputs[0].numpy()
            else:
                if self.extended_attention_mask:
                    replace_with_extended_mask(inputs, "causal", -1000)
                    inputs["attention_mask"] = inputs["attention_mask"].astype(
                        self.input_info["attention_mask"]["dtype"]
                    )
                outputs = session.run(None, {**inputs, **cache.get_kv_inputs()})
                logits = outputs[0]

            # Decide the next token using your preferred sampling strategy.
            # Sample with argmax (greedy search)
            if idx == 0:
                # logits for the first token are the logits at the last token of the prompt
                next_token_logits = np.take_along_axis(
                    logits, np_buffers["attention_mask"].sum(axis=1).reshape(batch_size, 1, 1) - 1, axis=1
                ).reshape(batch_size, -1)
            else:
                next_token_logits = logits[:, -1, :]
            next_tokens = np.argmax(next_token_logits, axis=-1)

            # Check if we previously reached EOS token id or if generated token id is EOS token id
            has_eos = has_eos | next_tokens == self.tokenizer.eos_token_id

            # Determine which new tokens to add to list of all token ids
            # Add EOS token ids for batch entries that ended early
            # (ragged batching scenario where some batch entries ended early and some haven't)
            tokens_to_add = np.where(~has_eos, next_tokens, self.tokenizer.eos_token_id).reshape(batch_size, 1)
            generated_tokens = np.concatenate([generated_tokens, tokens_to_add], axis=-1)

            # Return early if all batch entries have reached EOS token id
            if np.all(has_eos):
                break

            # Update inputs for next iteration
            if use_io_binding:
                if idx == 0:
                    inputs["input_ids"] = OrtValue.ortvalue_from_numpy(tokens_to_add, self.device, self.device_id)
                    if self.use_position_ids:
                        inputs["position_ids"] = OrtValue.ortvalue_from_numpy(
                            np_buffers["position_ids"], self.device, self.device_id
                        )
                else:
                    inputs["input_ids"].update_inplace(tokens_to_add)
                    if self.use_position_ids:
                        np_buffers["position_ids"] += 1
                        inputs["position_ids"].update_inplace(np_buffers["position_ids"])
            else:
                inputs["input_ids"] = tokens_to_add
                if self.use_position_ids:
                    if idx == 0:
                        inputs["position_ids"] = np_buffers["position_ids"]
                    else:
                        inputs["position_ids"] += 1

            # NOTE: we could use !has_eos instead of 1s in the attention updates but that is not necessary
            # since we don't care about subsequent tokens after EOS token id
            # NOTE: the attention over the past is technically not correct for gqa since gqa inserts the past
            # at seqlen_k instead of appending to the end of the past
            # it still works since the gqa node only cares about the sum of the attention mask
            if cache_type == "static":
                if idx == 0 and not isinstance(cache, GQASharedCache):
                    # past is now max_cache_len and there is only one token in the input
                    # static shape: (batch_size, max_cache_len + 1, num_heads, head_dim)
                    # GQA already has static shape: (batch_size, max_cache_len, num_heads, head_dim)
                    np_buffers["attention_mask"] = np.concatenate(
                        [
                            np_buffers["attention_mask"],
                            np.zeros((batch_size, cache.max_cache_len - prompt_length + 1), dtype=np.int32),
                        ],
                        1,
                    )
                elif idx > 0:
                    # previous token becomes past
                    np_buffers["attention_mask"][:, valid_prompt_len + idx - 1] = 1
                # always attend to the last token
                np_buffers["attention_mask"][:, -1] = 1
            else:
                np_buffers["attention_mask"] = np.concatenate(
                    [np_buffers["attention_mask"], np.ones((batch_size, 1), dtype=np.int32)], 1
                )

            attention_mask = np_buffers["attention_mask"]
            if self.extended_attention_mask:
                attention_mask = replace_with_extended_mask(
                    {
                        "input_ids": inputs["input_ids"].numpy() if use_io_binding else inputs["input_ids"],
                        "attention_mask": attention_mask,
                    },
                    "causal",
                    -1000,
                )["attention_mask"].astype(self.input_info["attention_mask"]["dtype"])

            if not use_io_binding:
                inputs["attention_mask"] = attention_mask
            elif cache_type == "dynamic" or (idx == 0 and not isinstance(cache, GQASharedCache)):
                # only update attention mask for dynamic cache or first token for static cache (non-GQA)
                inputs["attention_mask"] = OrtValue.ortvalue_from_numpy(attention_mask, self.device, self.device_id)
            else:
                # GQA, or static during token generation
                inputs["attention_mask"].update_inplace(attention_mask)

            first_cache = outputs[1]
            if use_io_binding:
                first_cache = first_cache.numpy()
            # print(first_cache[:, 0, :, 0])
            # if idx == 2:
            #     sdcd

            # update cache
            cache.update(outputs[1:])
        if use_io_binding:
            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

        decoded_text = self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        if isinstance(prompt, (str, tuple)):
            return decoded_text[0]
        return decoded_text

    def get_initial_inputs(
        self,
        prompt: Union[Union[str, Tuple[str]], List[Union[str, Tuple[str]]]],
        template: Optional[str],
        use_io_binding: bool,
        cache_type: str,
        max_cache_len: int,
        cache_backend: str,
    ) -> Tuple[Dict[str, Union["NDArray", OrtValue]], Union["Cache", "IOBoundCache"]]:
        """Get the initial inputs and cache for the model."""
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # using right padding since gqa expects it
        # other attention types should not care about padding side as long as the attention mask is correct
        self.tokenizer.padding_side = "right"

        if template is not None:
            prompt = (
                apply_template(template, prompt)
                if isinstance(prompt, (str, tuple))
                else [apply_template(template, p) for p in prompt]
            )
        else:
            assert isinstance(prompt, str) or (
                isinstance(prompt, list) and all(isinstance(p, str) for p in prompt)
            ), "tuple prompts require a template"
        prompt = [prompt] if isinstance(prompt, (str, tuple)) else prompt

        if self.batch_size:
            assert len(prompt) == self.batch_size, f"Batch size mismatch: {len(prompt)} != {self.batch_size}"
        # padding
        if self.prompt_len:
            padding_args = {"padding": "max_length", "max_length": self.prompt_len}
        else:
            padding_args = {"padding": "longest"}
        # encode prompt
        encodings_dict = self.tokenizer(prompt, return_tensors="np", add_special_tokens=False, **padding_args)
        input_ids = encodings_dict["input_ids"].astype(self.input_info["input_ids"]["dtype"])
        batch_size, prompt_length = input_ids.shape
        attention_mask = encodings_dict["attention_mask"]
        if not self.extended_attention_mask:
            attention_mask.astype(self.input_info["attention_mask"]["dtype"])
        # print(input_ids)
        # print(attention_mask)

        cache = self.get_fresh_cache(
            batch_size,
            use_io_binding,
            cache_type,
            self.cache_len or max_cache_len,
            cache_backend,
            int(attention_mask.sum(axis=-1).max()),
        )
        if isinstance(cache, GQASharedCache):
            print(cache.max_cache_len, prompt_length)
            attention_mask = np.concatenate(
                [attention_mask, np.zeros((batch_size, cache.max_cache_len - prompt_length), dtype=np.int32)], 1
            )

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "attention_mask_2d": attention_mask.copy()}
        if isinstance(cache, GQASharedCache) and self.prompt_len:
            # prompt processing needs to attend to the whole prompt+padding
            inputs["attention_mask"][:, :prompt_length] = 1
        if self.extended_attention_mask:
            replace_with_extended_mask(inputs, "causal", -1000)
            inputs["attention_mask"] = inputs["attention_mask"].astype(self.input_info["attention_mask"]["dtype"])
        if self.use_position_ids:
            position_ids = np.arange(prompt_length, dtype=np.int64).reshape((1, prompt_length))
            position_ids = np.broadcast_to(position_ids, (batch_size, prompt_length))
            inputs["position_ids"] = position_ids.astype(self.input_info["position_ids"]["dtype"])
        if use_io_binding:
            inputs = {k: OrtValue.ortvalue_from_numpy(v, self.device, self.device_id) for k, v in inputs.items()}
        return inputs, cache

    def get_fresh_cache(
        self,
        batch_size: int,
        use_io_binding: bool,
        cache_type: str,
        max_cache_len: int,
        cache_backend: str,
        valid_prompt_len: int,
    ) -> Union["Cache", "IOBoundCache"]:
        """Get a fresh cache for the model."""
        # determine cache class
        cache_class = None
        if cache_type == "static":
            if self.has_gqa and use_io_binding:
                cache_class = GQASharedCache
            elif self.has_gqa:
                # TODO(jambayk): implement generic GQA static cache if needed
                # it is not the same as using the current static cache since GQA inserts
                # new kv at seqlen_k instead of appending to the end of the past
                raise ValueError("GQA model only supports static cache with IO binding.")
            elif use_io_binding:
                cache_class = StaticIOBoundCache
            else:
                cache_class = StaticCache
        elif cache_type == "dynamic":
            if use_io_binding:
                cache_class = DynamicIOBoundCache
            else:
                cache_class = DynamicCache
        else:
            raise ValueError(f"Invalid cache type: {cache_type}")

        kwargs = {"batch_size": batch_size}
        if use_io_binding:
            kwargs["device"] = self.device
            kwargs["device_id"] = self.device_id
            if cache_type == "static":
                kwargs["backend"] = cache_backend
        if cache_type == "static":
            kwargs["max_cache_len"] = max_cache_len
            kwargs["valid_prompt_len"] = valid_prompt_len

        return cache_class(**kwargs, **self.cache_info)


def apply_template(template: str, p: Union[str, Tuple[str]]) -> str:
    if isinstance(p, tuple):
        kwargs = {f"prompt_{i}": p[i] for i in range(len(p))}
    else:
        kwargs = {"prompt": p}
    return template.format(**kwargs)
