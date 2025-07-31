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

__version__ = "0.9.2"

# pylint: disable=C0413

# Import Python API functions
from olive.cli.api import (  # noqa: E402
    capture_onnx_graph,
    convert_adapters,
    extract_adapters,
    finetune,
    generate_adapter,
    generate_cost_model,
    quantize,
    tune_session_params,
)
from olive.engine.output import DeviceOutput, ModelOutput, WorkflowOutput  # noqa: E402
from olive.workflows import run  # noqa: E402

__all__ = [
    "DeviceOutput",
    "ModelOutput",
    "WorkflowOutput",
    # Python API functions
    "capture_onnx_graph",
    "convert_adapters",
    "extract_adapters",
    "finetune",
    "generate_adapter",
    "generate_cost_model",
    "quantize",
    "run",
    "tune_session_params",
]
