# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from typing import List, Union

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

DOMAIN = "olive.auto_fusion"


def get_env_path(var_name):
    if not os.environ.get(var_name):
        raise RuntimeError(f"{var_name} not set")

    return Path(os.environ[var_name])


def create_triton_kernel_name(ops: List[str], dtype: str) -> str:
    """Create the name for the triton kernel for the fused op."""
    return "_".join(["triton", dtype, *[op.lower() for op in ops]])


def create_custom_op_name(ops: List[str], dtype: str) -> str:
    """Create the name for the custom op for the fused op."""
    return "".join(["Triton", dtype.upper(), *ops])


def join_params(params: Union[List, str], joiner: str = ",\n    ", end: str = ",", default: str = "# Not Used") -> str:
    """Join params with joiner and end."""
    if not params:
        return default
    params = [params] if isinstance(params, str) else params
    return joiner.join(params) + end
