# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import re
from itertools import chain
from typing import Any, Dict, List, Optional, Union

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator


class KVCacheConfig(ConfigBase):
    num_hidden_layers: int
    num_attention_heads: int
    hidden_size: int

    ort_past_key_name: str = "past_key_values.<id>.key"
    ort_past_value_name: str = "past_key_values.<id>.value"
    ort_present_key_name: str = "present.<id>.key"
    ort_present_value_name: str = "present.<id>.value"

    # world_size is used for distributed model. If world_size > 1,
    # the number of heads should be divisible by world_size
    world_size: int = 1
    past_sequence_length: int = None

    batch_size: int = 0
    dtype: Optional[str] = "float32"
    shared_kv: bool = False
    # most of the case the shape of the past_kv is
    # [batch_size, num_heads, past_sequence_length, hidden_size/num_heads]
    # but in some cases, the past_sequence_length is not the 3rd dimension
    # [batch_size, past_sequence_length, num_heads, hidden_size/num_heads]
    sequence_length_idx: int = 2
    past_kv_dynamic_axis: Optional[Dict] = None
    present_kv_dynamic_axis: Optional[Dict] = None

    @validator("past_kv_dynamic_axis", always=True)
    def check_past_kv_dynamic_axis(cls, v, values):
        is_shared_kv = values.get("shared_kv")
        msl_idx = str(values.get("sequence_length_idx"))
        if v is None:
            v = (
                {"0": "batch_size", msl_idx: "max_sequence_length"}
                if is_shared_kv
                else {"0": "batch_size", msl_idx: "past_sequence_length"}
            )
        return v

    @validator("present_kv_dynamic_axis", always=True)
    def check_present_kv_dynamic_axis(cls, v, values):
        is_shared_kv = values.get("shared_kv")
        msl_idx = str(values.get("sequence_length_idx"))
        if v is None:
            v = (
                {"0": "batch_size", msl_idx: "max_sequence_length"}
                if is_shared_kv
                else {"0": "batch_size", msl_idx: "past_sequence_length + sequence_length"}
            )
        return v

    def _get_k_names(self, direction="inputs"):
        if direction == "inputs":
            return [self.ort_past_key_name.replace("<id>", str(i)) for i in range(self.num_hidden_layers)]
        else:
            return [self.ort_present_key_name.replace("<id>", str(i)) for i in range(self.num_hidden_layers)]

    def _get_v_names(self, direction="inputs"):
        if direction == "inputs":
            return [self.ort_past_value_name.replace("<id>", str(i)) for i in range(self.num_hidden_layers)]
        else:
            return [self.ort_present_value_name.replace("<id>", str(i)) for i in range(self.num_hidden_layers)]

    def _get_kv_names(self, direction="inputs"):
        return list(chain.from_iterable(zip(self._get_k_names(direction), self._get_v_names(direction))))

    def get_ort_past_kv_names(self):
        return self._get_kv_names("inputs")

    def get_ort_present_kv_names(self):
        return self._get_kv_names("outputs")

    def _get_kv_shape(self):
        return [
            self.batch_size,
            self.num_attention_heads // self.world_size,
            self.past_sequence_length,
            self.hidden_size // self.num_attention_heads,
        ]

    def get_input_names_shapes_types(self):
        input_names = self.get_ort_past_kv_names()
        input_shapes = [self._get_kv_shape()] * 2 * self.num_hidden_layers
        input_types = [self.dtype] * 2 * self.num_hidden_layers

        return input_names, input_shapes, input_types

    def get_output_names(self):
        return self.get_ort_present_kv_names()

    def get_dynamic_axes(self):
        dynamic_axis = {}
        for past_name in self.get_ort_past_kv_names():
            dynamic_axis[past_name] = self.past_kv_dynamic_axis

        for present_name in self.get_ort_present_kv_names():
            dynamic_axis[present_name] = self.present_kv_dynamic_axis
        return dynamic_axis

    def _convert_to_dynamic_shapes_required_format(self, dynamic_shapes, kv_dynamic_shapes):

        def _update_dynamic_format(the_kv_dynamic_axis, batch_info):
            past_kv_dynamic_shapes = {}
            for axis, value in past_kv_dynamic_axis.items():
                if axis == "0":
                    past_kv_dynamic_shapes[axis] = batch_info
                else:
                    # Dim needs to pass str.isidentifier()
                    fixed_value = re.sub(r"[^A-Za-z_]", "", value)
                    # dynamic_shapes requires: [axis_name, min_value, max_value]
                    # More detail can be found in olive/passes/onnx/conversion.py
                    # TODO(titaiwang): Better way than hard-coded value?
                    past_kv_dynamic_shapes[axis] = [fixed_value, 0, 99999]
            return past_kv_dynamic_shapes

        # share the batch info
        if isinstance(dynamic_shapes, dict) and (batch_info := dynamic_shapes.get("input_ids", {}).get("0")):
            new_kv_dynamic_shapes = []
            for past_kv_dynamic_axis in kv_dynamic_shapes["past_key_value"]:
                past_kv_dynamic_shapes = _update_dynamic_format(past_kv_dynamic_axis, batch_info)
                new_kv_dynamic_shapes.append(past_kv_dynamic_shapes)
            kv_dynamic_shapes["past_key_value"] = new_kv_dynamic_shapes
            return kv_dynamic_shapes
        batch_info = dynamic_shapes[0].get("0")
        new_kv_dynamic_shapes = []
        for past_kv_dynamic_axis in kv_dynamic_shapes:
            past_kv_dynamic_shapes = _update_dynamic_format(past_kv_dynamic_axis, batch_info)
            new_kv_dynamic_shapes.append(past_kv_dynamic_shapes)
        return new_kv_dynamic_shapes

    def get_dynamic_shapes(
        self, dynamic_shapes: Optional[Union[List[Any], Dict[str, Any]]] = None
    ) -> Optional[Union[List[Any], Dict[str, Any]]]:
        """Get dynamic_shapes for past_key_values and present_kv.

        Unlike `dynamic_axes`, `dynamic_shapes` takes more than a string to set
        the dynamic axes. It requires a list of [axis_name, min_value, max_value].
        Also, if the same axis appears in different inputs, for example, the batch_size,
        we need to share the same dynamic shape by setting the same
        [axis_name, min_value, max_value].

        :param dynamic_shapes: the dynamic_shapes to convert
        :return: the converted dynamic_shapes
        """
        if dynamic_shapes is None:
            return None
        if isinstance(dynamic_shapes, dict):
            kv_dynamic_shapes = {}
            # Following the order of past_key_values and present
            kv_dynamic_shapes["past_key_value"] = []
            for _ in self.get_ort_past_kv_names():
                kv_dynamic_shapes["past_key_value"].append(self.past_kv_dynamic_axis)
            for _ in self.get_ort_present_kv_names():
                kv_dynamic_shapes["past_key_value"].append(self.present_kv_dynamic_axis)
            kv_dynamic_shapes = self._convert_to_dynamic_shapes_required_format(dynamic_shapes, kv_dynamic_shapes)
            dynamic_shapes.update(kv_dynamic_shapes)
            return dynamic_shapes
        if isinstance(dynamic_shapes, list):
            kv_dynamic_shapes = []
            for _ in self.get_ort_past_kv_names():
                kv_dynamic_shapes.append(self.past_kv_dynamic_axis)
            for _ in self.get_ort_present_kv_names():
                kv_dynamic_shapes.append(self.present_kv_dynamic_axis)
            kv_dynamic_shapes = self._convert_to_dynamic_shapes_required_format(dynamic_shapes, kv_dynamic_shapes)
            return [*dynamic_shapes, kv_dynamic_shapes]
        else:
            raise ValueError("dynamic_shapes should be a list or dict")
