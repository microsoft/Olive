# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.wizard import (
    DEVICE_CHOICES,
    PRECISION_CHOICES,
    _ask,
    _ask_select,
    build_calibration_args,
    prompt_calibration_source,
)

# ONNX operations
OP_OPTIMIZE = "optimize"
OP_QUANTIZE = "quantize"
OP_GRAPH_OPT = "graph_opt"
OP_CONVERT_PRECISION = "convert_precision"
OP_TUNE_SESSION = "tune_session"

# Quantization types
QUANT_STATIC = "static"
QUANT_DYNAMIC = "dynamic"
QUANT_BLOCKWISE_RTN = "blockwise_rtn"
QUANT_HQQ = "hqq"
QUANT_BNB = "bnb"


def run_onnx_flow(model_config):
    model_path = model_config.get("model_path", "")

    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Optimize model (auto-select best passes for target hardware)", value=OP_OPTIMIZE),
            questionary.Choice("Quantize", value=OP_QUANTIZE),
            questionary.Choice("Graph optimization", value=OP_GRAPH_OPT),
            questionary.Choice("Convert precision (FP32 \u2192 FP16)", value=OP_CONVERT_PRECISION),
            questionary.Choice("Tune session parameters", value=OP_TUNE_SESSION),
        ],
    )

    if operation == OP_OPTIMIZE:
        return _optimize_flow(model_path)
    elif operation == OP_QUANTIZE:
        return _quantize_flow(model_path)
    elif operation == OP_GRAPH_OPT:
        return _graph_opt_flow(model_path)
    elif operation == OP_CONVERT_PRECISION:
        return _convert_precision_flow(model_path)
    elif operation == OP_TUNE_SESSION:
        return _tune_session_flow(model_path)
    return {}


def _optimize_flow(model_path):
    provider = _ask(questionary.select("Select target device:", choices=DEVICE_CHOICES))
    precision = _ask(questionary.select("Select target precision:", choices=PRECISION_CHOICES))

    cmd = f"olive optimize -m {model_path} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _quantize_flow(model_path):
    quant_type = _ask(
        questionary.select(
            "Select quantization type:",
            choices=[
                questionary.Choice("Static Quantization (INT8) - requires calibration data", value=QUANT_STATIC),
                questionary.Choice("Dynamic Quantization (INT8) - no calibration needed", value=QUANT_DYNAMIC),
                questionary.Choice("Block-wise RTN (INT4) - no calibration needed", value=QUANT_BLOCKWISE_RTN),
                questionary.Choice("HQQ Quantization (INT4) - no calibration needed", value=QUANT_HQQ),
                questionary.Choice("BnB Quantization (FP4/NF4) - no calibration needed", value=QUANT_BNB),
            ],
        )
    )

    # Map to olive quantize CLI args
    quant_map = {
        QUANT_STATIC: {"implementation": "ort", "precision": "int8"},
        QUANT_DYNAMIC: {"implementation": "ort", "precision": "int8"},
        QUANT_BLOCKWISE_RTN: {"implementation": "ort", "precision": "int4"},
        QUANT_HQQ: {"implementation": "ort", "precision": "int4"},
        QUANT_BNB: {"implementation": "bnb", "precision": "nf4"},
    }

    params = quant_map[quant_type]
    cmd = (
        f"olive quantize -m {model_path} --precision {params['precision']} --implementation {params['implementation']}"
    )

    if quant_type == QUANT_DYNAMIC:
        cmd += " --algorithm rtn"

    # Calibration for static quantization
    if quant_type == QUANT_STATIC:
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
