# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Dict, List, Union

from olive.common.utils import hash_dict

TL_DTYPE_MAP = {
    "fp32": "tl.float32",
    "fp16": "tl.float16",
    "int32": "tl.int32",
    "int64": "tl.int64",
    "bool": "tl.bool",
    "bf16": "tl.bfloat16",
}

CPP_DTYPE_MAP = {
    "fp32": "float",
    "fp16": "Ort::Float16_t",
    "int32": "int32_t",
    "int64": "int64_t",
    "bool": "bool",
    "bf16": "Ort::BFloat16_t",
}

NP_DTYPE_REVERSE_MAP = {
    "float32": "fp32",
    "float16": "fp16",
    "int32": "int32",
    "int64": "int64",
    "bool": "bool",
    # bfloat16 is not supported in numpy
    # "bfloat16": "bf16",
}

DOMAIN = "olive.auto_fusion"

KERNEL_OUTPUT = "__fusion__output__"


def get_env_path(var_name):
    if not os.environ.get(var_name):
        raise RuntimeError(f"{var_name} not set")

    return Path(os.environ[var_name])


def hash_kernel_info(kernel_info: Dict) -> str:
    """Hash the kernel info to create a unique name for the custom op."""
    return hash_dict(kernel_info)[:8]


def create_triton_kernel_name(kernel_info: Dict) -> str:
    """Create the name for the triton kernel for the fused op."""
    return "_".join(["triton", *kernel_info["ops"], hash_kernel_info(kernel_info)])


def create_custom_op_name(kernel_info: Dict) -> str:
    """Create the name for the custom op for the fused op."""
    return "".join(["Triton", *kernel_info["ops"], "_", hash_kernel_info(kernel_info)])


def join_params(params: Union[List, str], joiner: str = ",\n    ", end: str = ",", default: str = "# Not Used") -> str:
    """Join params with joiner and end."""
    if not params:
        return default
    params = [params] if isinstance(params, str) else params
    return joiner.join(params) + end
