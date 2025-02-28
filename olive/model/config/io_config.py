# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from typing import Any, Dict, List, Union

from olive.common.config_utils import ConfigBase, get_the_flattened_and_tree_spec
from olive.common.hf.wrapper import ModelWrapper
from olive.common.pydantic_v1 import validator
from olive.model.config.kv_cache_config import KVCacheConfig


class IoConfig(ConfigBase):
    """IO config for model handler.

    For example, in stable diffusion, the config looks like:
    "io_config": {
        "input_names": [ "clip_input", "images" ],
        "output_names": [ "out_images", "has_nsfw_concepts" ],
        "dynamic_axes": {
            "clip_input": { "0": "batch", "1": "channels", "2": "height", "3": "width" },
            "images": { "0": "batch", "1": "height", "2": "width", "3": "channels" }
        },
        "dynamic_shapes": {
            "clip_input": { "0": "batch", "1": "channels",
                "2": "height", "3": "width" },
            "images": { "0": "batch", "1": "height",
                "2": "width", "3": "channels" }
        },
        "kv_cache": None
    }
    """

    # TODO(trajep): remove input names, shapes and types, turn to use olive dataset config.
    input_names: List[str]
    input_shapes: List[List[int]] = None
    input_types: List[str] = None
    output_names: List[str]
    output_shapes: List[List[int]] = None
    output_types: List[str] = None
    dynamic_axes: Dict[str, Dict[int, str]] = None
    # dynamic_shapes is different from dynamic_axes, it is nested.
    # We need to post-process its keys to int under onnx/conversion.py
    # for example, {"input_ids": {"0": "batch"}}
    dynamic_shapes: Union[List[Any], Dict[str, Any]] = None
    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference
    # even though we want the dimension to be a constant int.
    # We use a workaround here: first use dim_param like "1" to represent the dimension, and then
    # convert it to int in the onnx model.
    string_to_int_dim_params: List[str] = None
    # if False, skip kv_cache input
    # if True, use default KVCacheConfig
    # if KVCacheConfig, use the provided KVCacheConfig
    kv_cache: Union[bool, Dict[str, Any], KVCacheConfig] = False

    @validator("input_shapes", "input_types")
    def check_input_shapes(cls, v, values):
        if not v:
            return v

        if "input_names" not in values:
            raise ValueError("Invalid input_names")
        if len(v) != len(values["input_names"]):
            raise ValueError("input_names and input_shapes must have the same length")
        return v

    @validator("output_shapes", "output_types")
    def check_output_shapes(cls, v, values):
        if not v:
            return v

        if "output_names" not in values:
            raise ValueError("Invalid output_names")
        if len(v) != len(values["output_names"]):
            raise ValueError("output_names and output_shapes must have the same length")
        return v

    @validator("dynamic_axes")
    def convert_dynamic_axes(cls, v):
        if not v:
            return v

        dynamic_axes = v
        for k, value in dynamic_axes.items():
            dynamic_axes[k] = {int(kk): vv for kk, vv in value.items()}
        return dynamic_axes

    @validator("dynamic_shapes")
    def convert_dynamic_shapes(cls, v):
        if not v:
            return v

        flattened, tree_spec = get_the_flattened_and_tree_spec(v, leave_is_str=True)
        new_flattened = []
        for axes in flattened:
            if isinstance(axes, dict):
                new_flattened.append({int(kk): vv for kk, vv in axes.items()})
            else:
                new_flattened.append(axes)
        return tree_spec.unflatten(new_flattened)

    @validator("string_to_int_dim_params")
    def check_string_to_int_dim_params(cls, v):
        if not v:
            return v

        for dim_param in v:
            try:
                int(dim_param)
            except ValueError:
                raise ValueError(f"Invalid string_to_int_dim_params: {dim_param}. Must be castable to int.") from None
        return v

    def get_seq_len(self):
        if not self.input_shapes or not self.input_names:
            return 0
        for idx, name in enumerate(self.input_names):
            if name == "input_ids":
                return self.input_shapes[idx][1]  # pylint: disable=E1136
        return 0

    def get_past_seq_len(self):
        if not self.input_shapes or not self.input_names:
            return 0
        attention_mask_len, seq_len = 0, 0
        for idx, name in enumerate(self.input_names):
            if name == "attention_mask":
                attention_mask_len = self.input_shapes[idx][1]  # pylint: disable=E1136
            if name == "input_ids":
                seq_len = self.input_shapes[idx][1]  # pylint: disable=E1136
        return attention_mask_len - seq_len

    def get_batch_size(self):
        if not self.input_shapes or not self.input_names:
            return 1
        for idx, name in enumerate(self.input_names):
            if name == "input_ids":
                return self.input_shapes[idx][0]  # pylint: disable=E1136
        # if no input_ids, return 1
        return 1


def complete_kv_cache_with_model_attributes(kv_cache, model_attributes):
    model_wrapper = ModelWrapper(model_attributes)
    world_size = model_attributes.get("world_size", 1)
    kv_cache_obj = None
    if isinstance(kv_cache, bool) and kv_cache:
        kv_cache_obj = KVCacheConfig(
            num_hidden_layers=model_wrapper.num_hidden_layers,
            num_attention_heads=model_wrapper.num_attention_heads,
            hidden_size=model_wrapper.hidden_size,
            world_size=world_size,
        )
    elif isinstance(kv_cache, dict):
        kv_cache_dict = deepcopy(kv_cache)
        kv_cache_dict.update(
            {
                "num_hidden_layers": kv_cache.get("num_hidden_layers") or model_wrapper.num_hidden_layers,
                "num_attention_heads": kv_cache.get("num_attention_heads") or model_wrapper.num_attention_heads,
                "hidden_size": kv_cache.get("hidden_size") or model_wrapper.hidden_size,
                "world_size": kv_cache.get("world_size") or world_size,
            }
        )
        kv_cache_obj = KVCacheConfig.parse_obj(kv_cache_dict)
    elif isinstance(kv_cache, KVCacheConfig):
        # as num_hidden_layers, num_attention_heads, hidden_size are required for kv_cache
        # there is no need to update them
        kv_cache_obj = kv_cache

    if not kv_cache_obj.num_hidden_layers or not kv_cache_obj.num_attention_heads or not kv_cache_obj.hidden_size:
        raise ValueError(
            "num_hidden_layers, num_attention_heads, and hidden_size cannot be 0 or None, they"
            "are required for kv_cache."
        )
    return kv_cache_obj


def extend_io_config_with_kv_cache(io_config, kv_cache_config: KVCacheConfig):
    kv_cache_config.past_sequence_length = kv_cache_config.past_sequence_length or io_config.get_past_seq_len()
    kv_cache_config.batch_size = kv_cache_config.batch_size or io_config.get_batch_size()

    kv_names, kv_shapes, kv_types = kv_cache_config.get_input_names_shapes_types()
    output_names = kv_cache_config.get_output_names()
    dynamic_axes = deepcopy(io_config.dynamic_axes or {})
    dynamic_axes.update(kv_cache_config.get_dynamic_axes())
    dynamic_shapes = deepcopy(io_config.dynamic_shapes or {})
    dynamic_shapes.update(kv_cache_config.get_dynamic_shapes())
    return IoConfig(
        input_names=(io_config.input_names or []) + kv_names,
        input_shapes=(io_config.input_shapes or []) + kv_shapes,
        input_types=(io_config.input_types or []) + kv_types,
        output_names=(io_config.output_names or []) + output_names,
        output_shapes=io_config.output_shapes,  # ignore kv_cache output shapes
        output_types=io_config.output_types,  # ignore kv_cache output types
        dynamic_axes=dynamic_axes,
        dynamic_shapes=dynamic_shapes,
        kv_cache=kv_cache_config,
    )


def is_io_config_static(config: Union[IoConfig, Dict]):
    if isinstance(config, IoConfig):
        config = config.dict()
    if not config.get("input_shapes"):
        return False
    return all(all(isinstance(dim, int) for dim in shape) for shape in config["input_shapes"])


def is_kv_cache_required(input_past_kv_list, io_config: IoConfig):
    if io_config.kv_cache:
        return True
    # In the case of dynamo exporting, user do not need to provide input names
    # for past_key_values in huggingface pytorch model, when exported to onnx
    # and after PhiOnnxModel optimization, the pass_key_value will be extended as
    # `past_(key|value)_(hidden_layer_num)` pattern (phi2 case)
    # https://github.com/Microsoft/onnxruntime/blob/f53d2c2465d81cdb4e14c7241eab327184192c88/onnxruntime/python/tools/transformers/onnx_model_phi.py#L845C1-L845C31
    hidden_layer_num = len(input_past_kv_list)
    # possible kv-related variables which might be provided by the user
    # past_key_0, past_value_0, ...
    # past_key_values.0.key, past_key_values.0.value, ...
    # please keep adding more patterns if necessary
    possible_kv_names_templates = [
        ["past_key_<id>", "past_value_<id>"],
        # can be used to match past_key_values.0.key, past_key_values.0.value...
        # and past_key_values.0, past_key_values.0... at the same time
        ["past_key_values.<id>"],
    ]
    possible_kv_names_group = [[] for _ in range(len(possible_kv_names_templates))]

    for j, kv_template in enumerate(possible_kv_names_templates):
        for kv_str in kv_template:
            possible_kv_names_group[j].extend(kv_str.replace("<id>", str(i)) for i in range(hidden_layer_num))

    for kv_name_group in possible_kv_names_group:
        if all(any(kv_str in input_name for input_name in io_config.input_names) for kv_str in kv_name_group):
            return True
    return False
