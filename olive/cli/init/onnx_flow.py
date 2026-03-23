# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.helpers import (
    _ask,
    _ask_select,
    _device_choices,
    _precision_choices,
    build_calibration_args,
    prompt_calibration_source,
)
from olive.common.utils import StrEnumBase


class OnnxOperation(StrEnumBase):
    """ONNX operations."""

    OPTIMIZE = "optimize"
    QUANTIZE = "quantize"
    GRAPH_OPT = "graph_opt"
    CONVERT_PRECISION = "convert_precision"
    TUNE_SESSION = "tune_session"


class QuantizationType(StrEnumBase):
    """Quantization types."""

    STATIC = "static"
    DYNAMIC = "dynamic"
    BLOCKWISE_RTN = "blockwise_rtn"
    HQQ = "hqq"
    BNB = "bnb"


def run_onnx_flow(model_config):
    model_path = model_config.get("model_path", "")

    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice(
                "Optimize model (auto-select best passes for target hardware)", value=OnnxOperation.OPTIMIZE
            ),
            questionary.Choice("Quantize", value=OnnxOperation.QUANTIZE),
            questionary.Choice("Graph optimization", value=OnnxOperation.GRAPH_OPT),
            questionary.Choice("Convert precision (FP32 \u2192 FP16)", value=OnnxOperation.CONVERT_PRECISION),
            questionary.Choice("Tune session parameters", value=OnnxOperation.TUNE_SESSION),
        ],
    )

    if operation == OnnxOperation.OPTIMIZE:
        return _optimize_flow(model_path)
    elif operation == OnnxOperation.QUANTIZE:
        return _quantize_flow(model_path)
    elif operation == OnnxOperation.GRAPH_OPT:
        return _graph_opt_flow(model_path)
    elif operation == OnnxOperation.CONVERT_PRECISION:
        return _convert_precision_flow(model_path)
    elif operation == OnnxOperation.TUNE_SESSION:
        return _tune_session_flow(model_path)
    return {}


def _optimize_flow(model_path):
    provider = _ask(questionary.select("Select target device:", choices=_device_choices()))
    precision = _ask(questionary.select("Select target precision:", choices=_precision_choices()))

    cmd = f"olive optimize -m {model_path} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _quantize_flow(model_path):
    quant_type = _ask(
        questionary.select(
            "Select quantization type:",
            choices=[
                questionary.Choice(
                    "Static Quantization (INT8) - requires calibration data", value=QuantizationType.STATIC
                ),
                questionary.Choice(
                    "Dynamic Quantization (INT8) - no calibration needed", value=QuantizationType.DYNAMIC
                ),
                questionary.Choice(
                    "Block-wise RTN (INT4) - no calibration needed", value=QuantizationType.BLOCKWISE_RTN
                ),
                questionary.Choice("HQQ Quantization (INT4) - no calibration needed", value=QuantizationType.HQQ),
                questionary.Choice("BnB Quantization (FP4/NF4) - no calibration needed", value=QuantizationType.BNB),
            ],
        )
    )

    # Map to olive quantize CLI args
    quant_map = {
        QuantizationType.STATIC: {"implementation": "ort", "precision": "int8"},
        QuantizationType.DYNAMIC: {"implementation": "ort", "precision": "int8"},
        QuantizationType.BLOCKWISE_RTN: {"implementation": "ort", "precision": "int4"},
        QuantizationType.HQQ: {"implementation": "ort", "precision": "int4"},
        QuantizationType.BNB: {"implementation": "bnb", "precision": "nf4"},
    }

    params = quant_map[quant_type]
    cmd = (
        f"olive quantize -m {model_path} --precision {params['precision']} --implementation {params['implementation']}"
    )

    if quant_type == QuantizationType.DYNAMIC:
        cmd += " --algorithm rtn"

    # Calibration for static quantization
    if quant_type == QuantizationType.STATIC:
        calib = prompt_calibration_source()
        if calib:
            cmd += build_calibration_args(calib)

    return {"command": cmd}


def _graph_opt_flow(model_path):
    cmd = f"olive optimize -m {model_path} --precision fp32"
    return {"command": cmd}


def _convert_precision_flow(model_path):
    cmd = f"olive run-pass --pass-name OnnxFloatToFloat16 -m {model_path}"
    return {"command": cmd}


def _tune_session_flow(model_path):
    device = _ask(
        questionary.select(
            "Select target device:",
            choices=[
                questionary.Choice("CPU", value="cpu"),
                questionary.Choice("GPU", value="gpu"),
            ],
        )
    )

    providers = _ask(
        questionary.checkbox(
            "Select execution providers:",
            choices=[
                questionary.Choice("CPUExecutionProvider", value="CPUExecutionProvider", checked=(device == "cpu")),
                questionary.Choice("CUDAExecutionProvider", value="CUDAExecutionProvider", checked=(device == "gpu")),
                questionary.Choice("TensorrtExecutionProvider", value="TensorrtExecutionProvider"),
            ],
            instruction="(Space to toggle, Enter to confirm)",
        )
    )

    cmd = f"olive tune-session-params -m {model_path} --device {device}"
    if providers:
        cmd += " --providers_list " + " ".join(providers)

    cpu_cores = _ask(questionary.text("CPU cores for thread tuning (optional, press Enter to skip):", default=""))
    if cpu_cores:
        cmd += f" --cpu_cores {cpu_cores}"

    io_bind = _ask(questionary.confirm("Enable IO binding?", default=False))
    if io_bind:
        cmd += " --io_bind"

    enable_cuda_graph = _ask(questionary.confirm("Enable CUDA graph?", default=False))
    if enable_cuda_graph:
        cmd += " --enable_cuda_graph"

    return {"command": cmd}
