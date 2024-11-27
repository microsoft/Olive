# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.hardware.accelerator import (
    DEFAULT_CPU_ACCELERATOR,
    DEFAULT_GPU_CUDA_ACCELERATOR,
    DEFAULT_GPU_TRT_ACCELERATOR,
    AcceleratorLookup,
    AcceleratorSpec,
    Device,
)

__all__ = [
    "DEFAULT_CPU_ACCELERATOR",
    "DEFAULT_GPU_CUDA_ACCELERATOR",
    "DEFAULT_GPU_TRT_ACCELERATOR",
    "AcceleratorLookup",
    "AcceleratorSpec",
    "Device",
]
