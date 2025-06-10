# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import sys

_logger = logging.getLogger(__name__)
_logger.setLevel(logging.INFO)

_sc = logging.StreamHandler(stream=sys.stdout)
_formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [%(filename)s:%(lineno)d:%(funcName)s] %(message)s")
_sc.setFormatter(_formatter)
_logger.addHandler(_sc)
_logger.propagate = False

__version__ = "0.10.0.dev0"

# pylint: disable=C0413

from olive.engine.output import DeviceOutput, ModelOutput, WorkflowOutput  # noqa: E402

# Import Python API functions
from olive.api import (  # noqa: E402
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
    "DeviceOutput", 
    "ModelOutput", 
    "WorkflowOutput",
    # Python API functions
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
