# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Tuple, Union

import numpy as np
import onnx
from kv_cache_utils import DynamicCache, DynamicIOBoundCache, GQASharedCache, StaticCache, StaticIOBoundCache
from onnxruntime import InferenceSession, OrtValue, SessionOptions
from transformers import PreTrainedTokenizer

if TYPE_CHECKING:
    from kv_cache_utils import Cache, IOBoundCache
    from numpy.typing import NDArray


class AdapterMode(Enum):
    """Enum for adapter modes."""

    inputs = "inputs"
    initializers = "initializers"


class Generator:
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
        self.adapter_mode = adapter_mode

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

        # infer past names, dtype and dims
        self.cache_info = {"past_names": [], "present_names": [], "dtype": None, "num_kv_heads": None, "head_dim": None}
        session = self.sessions[self.adapters[self.default_adapter]["session"]]
        for i in session.get_inputs():
            if ".key" not in i.name and ".value" not in i.name:
                continue
            if not self.cache_info["past_names"]:
                self.cache_info["dtype"] = "float16" if i.type == "tensor(float16)" else "float32"
                self.cache_info["num_kv_heads"] = i.shape[1]
                self.cache_info["head_dim"] = i.shape[3]
            self.cache_info["past_names"].append(i.name)

        # infer present names
        for i in session.get_outputs():
            if ".key" not in i.name and ".value" not in i.name:
                continue
            self.cache_info["present_names"].append(i.name)

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
            sessions["default"] = InferenceSession(model_path, providers=providers, provider_options=provider_options)
            for name, info in adapter_info.items():
                adapters[name] = {
                    "session": "default",
                    "template": info.get("template"),
                }
                if info.get("weights") is not None:
                    np_weights = dict(np.load(info["weights"])) if isinstance(info["weights"], str) else info["weights"]
                    # TODO(jambayk): provide an option to lazy load the ortvalues if needed
                    # for example, there is not enough memory to load all adapters at once
                    ort_values = {k: OrtValue.ortvalue_from_numpy(v, device, device_id) for k, v in np_weights.items()}
                    adapters[name].update({"numpy_inputs": np_weights, "ortvalue_inputs": ort_values})
        else:
            # create a session for each adapter
            for name, info in adapter_info.items():
                # load the initializers and create the session
                initializer_names = []
                initializer_values = []
                if info.get("weights") is not None:
                    np_weights = dict(np.load(info["weights"])) if isinstance(info["weights"], str) else info["weights"]
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
        self, name: str = None, use_io_binding: bool = False
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
        prompt: Union[str, List[str]],
        adapter: str = None,
        max_gen_len: int = 128,
        use_io_binding: bool = False,
        cache_type: str = "dynamic",
        max_cache_len: int = 1024,
        cache_backend: str = "ort",
    ) -> List[str]:
        """Generate text from the model given a prompt.

        :param prompt: The prompt to generate text from. Can be a string or a list of strings.
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
            "attention_mask": (inputs["attention_mask"].numpy() if use_io_binding else inputs["attention_mask"].copy())
        }
        np_buffers["position_ids"] = np_buffers["attention_mask"].sum(axis=1).reshape(batch_size, 1)
        for idx in range(max_gen_len):
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
                    inputs["position_ids"] = OrtValue.ortvalue_from_numpy(
                        np_buffers["position_ids"], self.device, self.device_id
                    )
                else:
                    inputs["input_ids"].update_inplace(tokens_to_add)
                    np_buffers["position_ids"] += 1
                    inputs["position_ids"].update_inplace(np_buffers["position_ids"])
            else:
                inputs["input_ids"] = tokens_to_add
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

            # update cache
            cache.update(outputs[1:])
        if use_io_binding:
            io_binding.clear_binding_inputs()
            io_binding.clear_binding_outputs()

        return self.tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    def get_initial_inputs(
        self,
        prompt: Union[str, List[str]],
        template: str,
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
                template.format(prompt=prompt)
                if isinstance(prompt, str)
                else [template.format(prompt=p) for p in prompt]
            )

        encodings_dict = self.tokenizer(prompt, return_tensors="np", padding=True)
        input_ids = encodings_dict["input_ids"]
        batch_size, prompt_length = input_ids.shape
        attention_mask = encodings_dict["attention_mask"]

        position_ids = np.arange(prompt_length, dtype=np.int64).reshape((1, prompt_length))
        position_ids = np.broadcast_to(position_ids, (batch_size, prompt_length))

        cache = self.get_fresh_cache(batch_size, use_io_binding, cache_type, max_cache_len, cache_backend)
        if isinstance(cache, GQASharedCache):
            attention_mask = np.concatenate(
                [attention_mask, np.zeros((batch_size, cache.max_cache_len - prompt_length), dtype=np.int32)], 1
            )

        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
        }
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
