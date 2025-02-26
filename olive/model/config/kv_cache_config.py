# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from itertools import chain
from typing import Dict, Optional

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

    def get_dynamic_shapes(self):
        dynamic_shapes = {}
        past_kv_names = self.get_ort_past_kv_names()
        dynamic_shapes["past_key_values"] = [
            [self.past_kv_dynamic_axis, self.past_kv_dynamic_axis] for _ in range(0, len(past_kv_names), 2)
        ]
        return dynamic_shapes
