# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

PROVIDER_DOCKERFILE_MAPPING = {
    "CPUExecutionProvider": "Dockerfile.cpu",
    "CUDAExecutionProvider": "Dockerfile.gpu",
    "TensorrtExecutionProvider": "Dockerfile.gpu",
    "OpenVINOExecutionProvider": "Dockerfile.openvino",
}

PROVIDER_PACKAGE_MAPPING = {
    "CPUExecutionProvider": ("onnxruntime", "ort-nightly"),
    "CUDAExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
    "TensorrtExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
    "ROCMExecutionProvider": ("onnxruntime-gpu", "ort-nightly-gpu"),
    "OpenVINOExecutionProvider": ("onnxruntime-openvino", None),
    "DmlExecutionProvider": ("onnxruntime-directml", "ort-nightly-directml"),
}

DEVICE_TO_EXECUTION_PROVIDERS = {
    "cpu": {"CPUExecutionProvider", "OpenVINOExecutionProvider"},
    "gpu": {
        "DmlExecutionProvider",
        "CUDAExecutionProvider",
        "ROCMExecutionProvider",
        "MIGraphXExecutionProvider",
        "TensorrtExecutionProvider",
        "OpenVINOExecutionProvider",
        "JsExecutionProvider",
    },
    "npu": {"DmlExecutionProvider", "QNNExecutionProvider", "VitisAIExecutionProvider", "OpenVINOExecutionProvider"},
}
