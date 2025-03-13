# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
from kv_cache_utils import DynamicCache, DynamicIOBoundCache, GQASharedCache, StaticCache, StaticIOBoundCache
from onnxruntime import InferenceSession, OrtValue, SessionOptions, set_default_logger_severity
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from olive.common.utils import StrEnumBase, load_weights

if TYPE_CHECKING:
    from kv_cache_utils import Cache, IOBoundCache
    from numpy.typing import NDArray
    from onnx import ValueInfoProto


class AdapterMode(StrEnumBase):
    """Enum for adapter modes."""

    inputs = "inputs"
    initializers = "initializers"


# ort logs warnings about default initializers
set_default_logger_severity(3)


class ORTGenerator:
    def __init__(
        self,
        model_path: str,
        tokenizer: PreTrainedTokenizer,
        execution_provider: str,
        device_id: int = 0,
        adapters: Dict[str, Any] = None,
        adapter_mode: AdapterMode = AdapterMode.inputs,
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
        :param adapter_mode: The mode to use for the adapters. Can be "inputs" or "initializers".
        """
        self.model_path = model_path
        self.tokenizer = tokenizer
        self.execution_provider = execution_provider
        self.device_id = device_id
        self.adapter_info = adapters or {"default": {"weights": None, "template": None}}
        self.adapter_mode = AdapterMode(adapter_mode)

        # Determine attention type
        self.attn_type = "default"
        model_proto = onnx.load(self.model_path, load_external_data=False)
        for node in model_proto.graph.node:
            if node.op_type == "GroupQueryAttention":
                self.attn_type = "gqa"
                break
            if node.op_type == "MultiHeadAttention":
                self.attn_type = "mha"
                break
        # Get io info
        self.input_info = {}
        self.cache_info = {"past_names": [], "present_names": [], "dtype": None, "num_kv_heads": None, "head_dim": None}
        for i in model_proto.graph.input:
            if ".key" not in i.name and ".value" not in i.name:
                self.input_info[i.name] = self.get_io_type_shape(i)
                continue
            if not self.cache_info["past_names"]:
                past_info = self.get_io_type_shape(i)
                self.cache_info["dtype"] = past_info["dtype"]
                self.cache_info["num_kv_heads"] = past_info["shape"][1]
                self.cache_info["head_dim"] = past_info["shape"][3]
            self.cache_info["past_names"].append(i.name)
        for i in model_proto.graph.output:
            if ".key" not in i.name and ".value" not in i.name:
                continue
            self.cache_info["present_names"].append(i.name)
        del model_proto

        # Determine device to use for io binding
        # NOTE: QNNExecutionProvider does not support IO binding but keep it for future compatibility
        ep_to_device_map = {
            "CPUExecutionProvider": "cpu",
            "CUDAExecutionProvider": "cuda",
            "DmlExecutionProvider": "dml",
            "QNNExecutionProvider": "qnn",
        }
        self.device = ep_to_device_map[self.execution_provider]

        # create the session
        self.sessions, self.adapters = self.prepare_sessions(
            self.model_path, self.execution_provider, self.device, self.device_id, self.adapter_info, self.adapter_mode
        )
        self.default_adapter = next(iter(self.adapters.keys()))

    @staticmethod
    def get_io_type_shape(io: "ValueInfoProto") -> Dict:
        """Get the type and shape of an input/output."""
        tensor_type = io.type.tensor_type
        if tensor_type.elem_type == 0:
            # sequence type
            # TODO(jambayk): add support for different types
            # refer to https://github.com/lutzroeder/netron/blob/main/source/onnx.js#L1424
            tensor_type = io.type.sequence_type.elem_type.tensor_type
        data_type = onnx.helper.tensor_dtype_to_np_dtype(tensor_type.elem_type).name
        shape = [dim.dim_param if dim.dim_param else dim.dim_value for dim in tensor_type.shape.dim]
        return {
            "dtype": data_type,
            "shape": shape,
        }

    @property
    def use_position_ids(self) -> bool:
        """Whether the model has position ids."""
        return "position_ids" in self.input_info

    @property
    def use_past_seq_len(self) -> bool:
        """Whether the model has past sequence length."""
        return "past_seq_len" in self.input_info

    @staticmethod
    def prepare_sessions(
        model_path: str,
        execution_provider: str,
        device: str,
        device_id: int,
        adapter_info: Dict[str, Any],
        adapter_mode: AdapterMode,
    ) -> Tuple[Dict[str, InferenceSession], Dict[str, Dict[str, Any]]]:
        """Prepare the sessions and adapters for the model.

        :param model_path: Path to the model.
        :param execution_provider: The execution provider to use.
        :param device: The device to use.
        :param device_id: The device id to use.
        :param adapter_info: Dictionary of adapter information. Each key is the adapter name and the value is a
            dictionary with the following keys
            - "weights": Path to the weights file or dictionary of weights.
            - "template": The template to use for the adapter.
        :param adapter_mode: The mode to use for the adapters. Can be "inputs" or "initializers".
        :return: Tuple of sessions and adapters.
            Sessions is a dictionary of session names to InferenceSession objects.
            Adapters is a dictionary of adapter names to dictionaries with the following keys
            - "session": The session name to use for the adapter.
            - "template": The template to use for the adapter.
            - "numpy_inputs": The numpy inputs for the adapter if adapter_mode is "inputs".
            - "ortvalue_inputs": The OrtValue inputs for the adapter if adapter_mode is "inputs".
        """
        sessions = {}
        adapters = {}
        providers = [execution_provider]
        provider_options = [{"device_id": device_id}] if device in {"cuda", "dml"} else None
        if adapter_mode == AdapterMode.inputs:
            # there is only one session
            # TODO(jambayk): test and enable graph for cuda and dml
            sessions["default"] = InferenceSession(model_path, providers=providers, provider_options=provider_options)
            for name, info in adapter_info.items():
                adapters[name] = {
                    "session": "default",
                    "template": info.get("template"),
                }
                if info.get("weights") is not None:
                    np_weights = load_weights(info["weights"]) if isinstance(info["weights"], str) else info["weights"]
                    # TODO(jambayk): provide an option to lazy load the ortvalues if needed
                    # for example, there is not enough memory to load all adapters at once
                    if execution_provider != "QNNExecutionProvider":
                        ort_values = {
                            k: OrtValue.ortvalue_from_numpy(v, device, device_id) for k, v in np_weights.items()
                        }
                    else:
                        ort_values = None
                    adapters[name].update({"numpy_inputs": np_weights, "ortvalue_inputs": ort_values})
        else:
            # create a session for each adapter
            for name, info in adapter_info.items():
                # load the initializers and create the session
                initializer_names = []
                initializer_values = []
                if info.get("weights") is not None:
                    np_weights = load_weights(info["weights"]) if isinstance(info["weights"], str) else info["weights"]
                    for i_name, value in np_weights.items():
                        initializer_names.append(i_name)
                        initializer_values.append(OrtValue.ortvalue_from_numpy(value))

                session_options = SessionOptions()
                session_options.add_external_initializers(initializer_names, initializer_values)

                sessions[name] = InferenceSession(
                    model_path, providers=providers, provider_options=provider_options, sess_options=session_options
                )
                adapters[name] = {
                    "session": name,
                    "template": info.get("template"),
                }
        return sessions, adapters

    def get_adapter(
        self, name: Optional[str] = None, use_io_binding: bool = False
    ) -> Tuple[InferenceSession, str, Dict[str, Any]]:
        """Get the session, template and inputs for the specified adapter.

        :param name: Name of the adapter to use. If None, the default adapter is used.
        :param use_io_binding: Whether to use IO binding for the adapter.
        :return: Tuple of session, template and inputs.
            The adapter_inputs are either numpy arrays or OrtValues depending on use_io_binding.
        """
        if name is None:
            name = self.default_adapter
        elif name not in self.adapters:
            raise ValueError(f"Adapter {name} not found.")

        session = self.sessions[self.adapters[name]["session"]]
        template = self.adapters[name]["template"]
        inputs = (
            self.adapters[name].get("ortvalue_inputs") if use_io_binding else self.adapters[name].get("numpy_inputs")
        )
        return session, template, inputs

    def generate(
        self,
        prompt: Union[Union[str, Tuple[str]], List[Union[str, Tuple[str]]]],
        adapter: Optional[str] = None,
        max_gen_len: int = 128,
        use_io_binding: bool = False,
        cache_type: str = "dynamic",
        max_cache_len: int = 1024,
        cache_backend: str = "ort",
    ) -> Union[str, List[str]]:
        """Generate text from the model given a prompt.

        :param prompt: The prompt to generate text from. Can be a string/string-tuples or a list.
        :param adapter: The adapter to use for generation. If None, the default adapter is used.
        :param max_gen_len: The maximum length of the generated text.
        :param use_io_binding: Whether to use IO binding for the generation.
        :param cache_type: The type of cache to use for the generation. Can be "static" or "dynamic".
        :param max_cache_len: The maximum length of the cache for static cache.
        :param cache_backend: The backend to use for static cache. Can be "ort" or "torch".
        :return: The generated text.
        """
        if use_io_binding and self.execution_provider == "QNNExecutionProvider":
            raise ValueError("QNNExecutionProvider does not support IO binding.")

        # get session, template and adapter inputs
        # if adapter mode is inputs, the adapter input is dictionary of adapter weights
        session, template, adapter_inputs = self.get_adapter(adapter, use_io_binding)
        io_binding = session.io_binding() if use_io_binding else None

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
            "attention_mask": inputs["attention_mask"].numpy() if use_io_binding else inputs["attention_mask"].copy()
        }
        if self.use_position_ids:
            np_buffers["position_ids"] = (
                np_buffers["attention_mask"]
                .sum(axis=1)
                .reshape(batch_size, 1)
                .astype(self.input_info["position_ids"]["dtype"])
            )

        for idx in tqdm(range(max_gen_len)):
            used_inputs = {k: v for k, v in inputs.items() if k in self.input_info}
            if use_io_binding:
                if idx < 2:
                    # need to bind logits twice, once for prompt processing and once for token generation
                    # shapes: (batch_size, prompt_length, vocab_size) and (batch_size, 1, vocab_size)
                    io_binding.bind_output("logits", self.device, self.device_id)
                for k, v in used_inputs.items():
                    io_binding.bind_ortvalue_input(k, v)
                cache.bind_kv_io(io_binding)

                io_binding.synchronize_inputs()
                session.run_with_iobinding(io_binding)
                io_binding.synchronize_outputs()

                outputs = io_binding.get_outputs()
                logits = outputs[0].numpy()
            else:
                outputs = session.run(None, {**{used_inputs}, **cache.get_kv_inputs()})
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
                    np_buffers["attention_mask"][:, prompt_length + idx - 1] = 1
                # always attend to the last token
                np_buffers["attention_mask"][:, -1] = 1
            else:
                np_buffers["attention_mask"] = np.concatenate(
                    [np_buffers["attention_mask"], np.ones((batch_size, 1), dtype=np.int32)], 1
                )

            if not use_io_binding:
                inputs["attention_mask"] = np_buffers["attention_mask"]
            elif cache_type == "dynamic" or (idx == 0 and not isinstance(cache, GQASharedCache)):
                # only update attention mask for dynamic cache or first token for static cache (non-GQA)
                inputs["attention_mask"] = OrtValue.ortvalue_from_numpy(
                    np_buffers["attention_mask"], self.device, self.device_id
                )
            else:
                # GQA, or static during token generation
                inputs["attention_mask"].update_inplace(np_buffers["attention_mask"])

            if self.use_past_seq_len:
                past_seq_len = (np_buffers["attention_mask"].sum(-1, keepdims=True) - 1).astype(np.int32)
                total_seq_len = np.array(np_buffers["attention_mask"].shape[1], dtype=np.int32)
                if use_io_binding:
                    inputs["past_seq_len"].update_inplace(past_seq_len)
                    inputs["total_seq_len"].update_inplace(total_seq_len)
                else:
                    inputs["past_seq_len"] = past_seq_len
                    inputs["total_seq_len"] = total_seq_len

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

        encodings_dict = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = encodings_dict["input_ids"].astype(self.input_info["input_ids"]["dtype"])
        batch_size, prompt_length = input_ids.shape
        attention_mask = encodings_dict["attention_mask"]
        if "attention_mask" in self.input_info:
            attention_mask = attention_mask.astype(self.input_info["attention_mask"]["dtype"])

        cache = self.get_fresh_cache(batch_size, use_io_binding, cache_type, max_cache_len, cache_backend)
        if isinstance(cache, GQASharedCache):
            attention_mask = np.concatenate(
                [attention_mask, np.zeros((batch_size, cache.max_cache_len - prompt_length), dtype=np.int32)], 1
            )

        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if self.use_position_ids:
            position_ids = np.arange(prompt_length, dtype=np.int64).reshape((1, prompt_length))
            position_ids = np.broadcast_to(position_ids, (batch_size, prompt_length))
            inputs["position_ids"] = position_ids.astype(self.input_info["position_ids"]["dtype"])
        if self.use_past_seq_len:
            inputs["past_seq_len"] = (attention_mask.sum(-1, keepdims=True) - 1).astype(np.int32)
            inputs["total_seq_len"] = np.array(attention_mask.shape[1], dtype=np.int32)
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
    ) -> Union["Cache", "IOBoundCache"]:
        """Get a fresh cache for the model."""
        # determine cache class
        cache_class = None
        if cache_type == "static":
            if self.attn_type == "gqa" and use_io_binding:
                cache_class = GQASharedCache
            elif self.attn_type == "gqa":
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

        return cache_class(**kwargs, **self.cache_info)


def apply_template(template: str, p: Union[str, Tuple[str]]) -> str:
    if isinstance(p, tuple):
        kwargs = {f"prompt_{i}": p[i] for i in range(len(p))}
    else:
        kwargs = {"prompt": p}
    return template.format(**kwargs)
