# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import List

from pydantic import validator

from olive.common.config_utils import ConfigBase

logging_verbosity = {
    "debug": logging.DEBUG,
    "info": logging.INFO,
    "warning": logging.WARNING,
    "error": logging.ERROR,
    "critical": logging.CRITICAL,
}


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


class ConvertQuantizeConfig(ConfigBase):
    model: Path
    io_config: IOConfig
    quant_data: Path
    input_list_file: str = None
    tool: str
    convert_options: dict = None
    quantize_options: dict = None
    output_dir: Path = None
    output_name: str = None
    workspace: Path = None
    verbosity: str = "info"

    @validator("tool")
    def check_tool(cls, v):
        valid_tools = ["snpe", "openvino"]
        if v not in valid_tools:
            raise ValueError(f"tool must be one of {valid_tools}")
        return v

    @validator("input_list_file")
    def check_input_list_file(cls, v, values):
        if "quant_data" not in values:
            raise ValueError("Invalid quant_data")
        if v is not None:
            if values["quant_data"].glob(v) == []:
                raise ValueError(f"input_list_file {v} not found in {values['quant_data']}")
        return v

    @validator("verbosity")
    def check_verbosity(cls, v):
        if v not in logging_verbosity:
            raise ValueError(f"verbosity must be one of {logging_verbosity.keys()}")
        return v
