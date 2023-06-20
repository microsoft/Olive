# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List

from pydantic import validator

from olive.common.config_utils import ConfigBase


class IOConfig(ConfigBase):
    input_names: List[str]
    input_shapes: List[List[int]]
    output_names: List[str]
    output_shapes: List[List[int]]

    @validator("input_shapes")
    def check_input_shapes(cls, v, values):
        if "input_names" not in values:
            raise ValueError("Invalid input_names")
        if len(v) != len(values["input_names"]):
            raise ValueError("input_names and input_shapes must have the same length")
        return v

    @validator("output_shapes")
    def check_output_shapes(cls, v, values):
        if "output_names" not in values:
            raise ValueError("Invalid output_names")
        if len(v) != len(values["output_names"]):
            raise ValueError("output_names and output_shapes must have the same length")
        return v


class ToolConfig(ConfigBase):
    convert_options: dict = None
    quantize_options: dict = None


class ConvertQuantizeConfig(ConfigBase):
    model: Path
    io_config: IOConfig
    quant_data: Path
    input_dirs: List[str] = None
    input_order_file: str = None
    tools: Dict[str, ToolConfig]
    output_dir: Path = None
    log_severity_level: int = 1

    @validator("input_dirs")
    def check_input_dirs(cls, v, values):
        if "io_config" not in values:
            raise ValueError("Invalid io_config")

        if len(v) != len(values["io_config"].input_names):
            raise ValueError("input_dirs and input_names must have the same length")
        return v

    @validator("input_order_file")
    def check_input_order_file(cls, v, values):
        if "quant_data" not in values:
            raise ValueError("Invalid quant_data")

        if list(values["quant_data"].glob(v)) == []:
            raise ValueError(f"input_order_file {v} not found in {values['quant_data']}")
        return v
