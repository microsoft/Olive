# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Dict, List, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase


class IOConfig(ConfigBase):
    # TODO(trajep): remove input names, shapes and types, turn to use olive dataset config.
    input_names: List[str]
    input_shapes: List[List[int]] = None
    input_types: List[str] = None
    output_names: List[str]
    output_shapes: List[List[int]] = None
    output_types: List[str] = None
    dynamic_axes: Dict[str, Dict[int, str]] = None
    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference
    # even though we want the dimension to be a constant int.
    # We use a workaround here: first use dim_param like "1" to represent the dimension, and then
    # convert it to int in the onnx model.
    string_to_int_dim_params: List[str] = None

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
        for k, v in dynamic_axes.items():
            dynamic_axes[k] = {int(kk): vv for kk, vv in v.items()}
        return dynamic_axes

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


def is_io_config_static(config: Union[IOConfig, Dict]):
    if isinstance(config, IOConfig):
        config = config.dict()
    if not config["input_shapes"]:
        return False
    return all(all(isinstance(dim, int) for dim in shape) for shape in config["input_shapes"])
