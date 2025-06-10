# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Python API for Olive CLI commands.

This module provides Python functions that correspond to each CLI command,
allowing users to programmatically execute Olive workflows and receive
WorkflowOutput results containing ModelOutput instances.
"""

from olive.api.workflow import (
    auto_opt,
    capture_onnx,
    configure_qualcomm_sdk,
    convert_adapters,
    extract_adapters,
    finetune,
    generate_adapter,
    generate_cost_model,
    manage_aml_compute,
    quantize,
    run,
    session_params_tuning,
    shared_cache,
)

__all__ = [
    "auto_opt",
    "capture_onnx", 
    "configure_qualcomm_sdk",
    "convert_adapters",
    "extract_adapters",
    "finetune",
    "generate_adapter",
    "generate_cost_model",
    "manage_aml_compute",
    "quantize",
    "run",
    "session_params_tuning",
    "shared_cache",
]