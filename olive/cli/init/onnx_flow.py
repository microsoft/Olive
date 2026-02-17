# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import questionary

from olive.cli.init.wizard import _ask, _ask_select

DEVICE_CHOICES = [
    questionary.Choice("CPU", value="CPUExecutionProvider"),
    questionary.Choice("GPU (NVIDIA CUDA)", value="CUDAExecutionProvider"),
    questionary.Choice("NPU (Qualcomm QNN)", value="QNNExecutionProvider"),
    questionary.Choice("NPU (Intel OpenVINO)", value="OpenVINOExecutionProvider"),
    questionary.Choice("NPU (AMD Vitis AI)", value="VitisAIExecutionProvider"),
    questionary.Choice("WebGPU", value="WebGpuExecutionProvider"),
]

PRECISION_CHOICES = [
    questionary.Choice("INT4", value="int4"),
    questionary.Choice("INT8", value="int8"),
    questionary.Choice("FP16", value="fp16"),
    questionary.Choice("FP32", value="fp32"),
]


def run_onnx_flow(model_config):
    model_path = model_config.get("model_path", "")

    operation = _ask_select(
        "What do you want to do?",
        choices=[
            questionary.Choice("Optimize model (auto-select best passes for target hardware)", value="optimize"),
            questionary.Choice("Quantize", value="quantize"),
            questionary.Choice("Graph optimization", value="graph_opt"),
            questionary.Choice("Convert precision (FP32 \u2192 FP16)", value="convert_precision"),
            questionary.Choice("Tune session parameters", value="tune_session"),
        ],
    )

    if operation == "optimize":
        return _optimize_flow(model_path)
    elif operation == "quantize":
        return _quantize_flow(model_path)
    elif operation == "graph_opt":
        return _graph_opt_flow(model_path)
    elif operation == "convert_precision":
        return _convert_precision_flow(model_path)
    elif operation == "tune_session":
        return _tune_session_flow(model_path)
    return {}


def _optimize_flow(model_path):
    provider = _ask(questionary.select("Select target device:", choices=DEVICE_CHOICES))
    precision = _ask(questionary.select("Select target precision:", choices=PRECISION_CHOICES))

    cmd = f"olive optimize -m {model_path} --provider {provider} --precision {precision}"
    return {"command": cmd}


def _quantize_flow(model_path):
    quant_type = _ask(questionary.select(
        "Select quantization type:",
        choices=[
            questionary.Choice("Static Quantization (INT8) - requires calibration data", value="static"),
            questionary.Choice("Dynamic Quantization (INT8) - no calibration needed", value="dynamic"),
            questionary.Choice("Block-wise RTN (INT4) - no calibration needed", value="blockwise_rtn"),
            questionary.Choice("HQQ Quantization (INT4) - no calibration needed", value="hqq"),
            questionary.Choice("BnB Quantization (FP4/NF4) - no calibration needed", value="bnb"),
        ],
    ))

    # Map to olive quantize CLI args
    quant_map = {
        "static": {"implementation": "ort", "precision": "int8"},
        "dynamic": {"implementation": "ort", "precision": "int8"},
        "blockwise_rtn": {"implementation": "ort", "precision": "int4"},
        "hqq": {"implementation": "ort", "precision": "int4"},
        "bnb": {"implementation": "bnb", "precision": "nf4"},
    }

    params = quant_map[quant_type]
    cmd = f"olive quantize -m {model_path} --precision {params['precision']} --implementation {params['implementation']}"

    if quant_type == "dynamic":
        cmd += " --algorithm rtn"

    # Calibration for static quantization
    if quant_type == "static":
        calib = _prompt_calibration_data()
        if calib:
            cmd += calib

    return {"command": cmd}


def _graph_opt_flow(model_path):
    _ask(questionary.checkbox(
        "Select optimizations:",
        choices=[
            questionary.Choice("Peephole optimization (constant folding, dead code elimination)", value="peephole", checked=True),
            questionary.Choice("Transformer optimization (operator fusion for transformers)", value="transformer", checked=True),
        ],
    ))

    cmd = f"olive optimize -m {model_path} --precision fp32"
    return {"command": cmd}


def _convert_precision_flow(model_path):
    cmd = f"olive run-pass --pass-name OnnxFloatToFloat16 -m {model_path}"
    return {"command": cmd}


def _tune_session_flow(model_path):
    device = _ask(questionary.select(
        "Select target device:",
        choices=[
            questionary.Choice("CPU", value="cpu"),
            questionary.Choice("GPU", value="gpu"),
        ],
    ))

    providers = _ask(questionary.checkbox(
        "Select execution providers:",
        choices=[
            questionary.Choice("CPUExecutionProvider", value="CPUExecutionProvider", checked=(device == "cpu")),
            questionary.Choice("CUDAExecutionProvider", value="CUDAExecutionProvider", checked=(device == "gpu")),
            questionary.Choice("TensorrtExecutionProvider", value="TensorrtExecutionProvider"),
        ],
    ))

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


def _prompt_calibration_data():
    """Prompt for calibration data and return CLI args string."""
    source = _ask(questionary.select(
        "Calibration data source:",
        choices=[
            questionary.Choice("HuggingFace dataset", value="hf"),
            questionary.Choice("Local file", value="local"),
        ],
    ))

    if source == "hf":
        data_name = _ask(questionary.text("Dataset name:", default="Salesforce/wikitext"))
        subset = _ask(questionary.text("Subset:", default="wikitext-2-raw-v1"))
        split = _ask(questionary.text("Split:", default="train"))
        num_samples = _ask(questionary.text("Number of samples:", default="128"))

        result = f" -d {data_name}"
        if subset:
            result += f" --subset {subset}"
        result += f" --split {split} --max_samples {num_samples}"
        return result
    else:
        data_files = _ask(questionary.text("Data file path:"))
        return f" --data_files {data_files}"
