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

try:
    import onnxruntime as ort

    # pylint: disable=protected-access
    ort._get_available_providers = ort.get_available_providers

    def get_available_providers_winml():
        # pylint: disable=protected-access
        providers = ort._get_available_providers()
        extra_providers = {ep_device.ep_name for ep_device in ort.get_ep_devices()} - set(providers)
        return providers + list(extra_providers)

    ort.get_available_providers = get_available_providers_winml
except Exception:
    pass

# pylint: disable=C0413

# Import Python API functions
from olive.cli.api import (  # noqa: E402
    auto_opt,
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
    "auto_opt",
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
