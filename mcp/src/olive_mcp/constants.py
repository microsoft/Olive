# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import StrEnum
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VENV_BASE = Path.home() / ".olive-mcp" / "venvs"
OUTPUT_BASE = Path.home() / ".olive-mcp" / "outputs"
WORKER_PATH = Path(__file__).parent / "worker.py"

# Auto-purge venvs not used within this many days.
_VENV_MAX_AGE_DAYS = 14

# ---------------------------------------------------------------------------
# Command names
# ---------------------------------------------------------------------------


class Command(StrEnum):
    OPTIMIZE = "optimize"
    QUANTIZE = "quantize"
    FINETUNE = "finetune"
    CAPTURE_ONNX_GRAPH = "capture_onnx_graph"
    BENCHMARK = "benchmark"
    DIFFUSION_LORA = "diffusion_lora"
    EXPLORE_PASSES = "explore_passes"
    VALIDATE_CONFIG = "validate_config"
    RUN_CONFIG = "run_config"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class SupportedProvider(StrEnum):
    CPU = "CPUExecutionProvider"
    CUDA = "CUDAExecutionProvider"
    DML = "DmlExecutionProvider"
    OPENVINO = "OpenVINOExecutionProvider"
    TENSORRT = "TensorrtExecutionProvider"
    ROCM = "ROCMExecutionProvider"
    QNN = "QNNExecutionProvider"
    VITISAI = "VitisAIExecutionProvider"
    WEBGPU = "WebGpuExecutionProvider"
    NV_TENSORRT_RTX = "NvTensorRTRTXExecutionProvider"


class SupportedPrecision(StrEnum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    UINT4 = "uint4"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"


SUPPORTED_QUANT_ALGORITHMS = ["rtn", "gptq", "awq", "hqq"]

# Maps provider → olive-ai extras key for onnxruntime variant
PROVIDER_TO_EXTRAS = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "gpu",
    "TensorrtExecutionProvider": "gpu",
    "ROCMExecutionProvider": "gpu",
    "OpenVINOExecutionProvider": "openvino",
    "DmlExecutionProvider": "directml",
    "QNNExecutionProvider": "qnn",
}

# Maps provider → onnxruntime-genai variant (for ModelBuilder pass)
PROVIDER_TO_GENAI = {
    "CPUExecutionProvider": "onnxruntime-genai",
    "CUDAExecutionProvider": "onnxruntime-genai-cuda",
    "DmlExecutionProvider": "onnxruntime-genai-directml",
}
