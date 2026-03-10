# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VENV_BASE = Path.home() / ".olive-mcp" / "venvs"
OUTPUT_BASE = Path.home() / ".olive-mcp" / "outputs"
WORKER_PATH = Path(__file__).parent / "worker.py"

# Bump this when _resolve_packages logic changes to invalidate stale cached venvs.
_VENV_CACHE_VERSION = "v2"

# Auto-purge venvs not used within this many days.
_VENV_MAX_AGE_DAYS = 14

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_PROVIDERS = [
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "WebGpuExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
]

SUPPORTED_PRECISIONS = [
    "fp32",
    "fp16",
    "bf16",
    "int4",
    "int8",
    "int16",
    "int32",
    "uint4",
    "uint8",
    "uint16",
    "uint32",
]

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
