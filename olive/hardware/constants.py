# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.common.utils import StrEnumBase


class ExecutionProvider(StrEnumBase):
    CPUExecutionProvider = "CPUExecutionProvider"
    CUDAExecutionProvider = "CUDAExecutionProvider"
    DmlExecutionProvider = "DmlExecutionProvider"
    OpenVINOExecutionProvider = "OpenVINOExecutionProvider"
    TensorrtExecutionProvider = "TensorrtExecutionProvider"
    ROCMExecutionProvider = "ROCMExecutionProvider"
    MIGraphXExecutionProvider = "MIGraphXExecutionProvider"
    NvTensorRTRTXExecutionProvider = "NvTensorRTRTXExecutionProvider"
    JsExecutionProvider = "JsExecutionProvider"
    QNNExecutionProvider = "QNNExecutionProvider"
    VitisAIExecutionProvider = "VitisAIExecutionProvider"
    WebGpuExecutionProvider = "WebGpuExecutionProvider"


PROVIDER_DOCKERFILE_MAPPING = {
    ExecutionProvider.CPUExecutionProvider: "Dockerfile.cpu",
    ExecutionProvider.CUDAExecutionProvider: "Dockerfile.gpu",
    ExecutionProvider.TensorrtExecutionProvider: "Dockerfile.gpu",
    ExecutionProvider.OpenVINOExecutionProvider: "Dockerfile.openvino",
}

PROVIDER_PACKAGE_MAPPING = {
    ExecutionProvider.CPUExecutionProvider: "onnxruntime",
    ExecutionProvider.CUDAExecutionProvider: "onnxruntime-gpu",
    ExecutionProvider.TensorrtExecutionProvider: "onnxruntime-gpu",
    ExecutionProvider.ROCMExecutionProvider: "onnxruntime-gpu",
    ExecutionProvider.OpenVINOExecutionProvider: "onnxruntime-openvino",
    ExecutionProvider.DmlExecutionProvider: "onnxruntime-directml",
}

DEVICE_TO_EXECUTION_PROVIDERS = {
    "cpu": {ExecutionProvider.CPUExecutionProvider, ExecutionProvider.OpenVINOExecutionProvider},
    "gpu": {
        ExecutionProvider.DmlExecutionProvider,
        ExecutionProvider.CUDAExecutionProvider,
        ExecutionProvider.ROCMExecutionProvider,
        ExecutionProvider.MIGraphXExecutionProvider,
        ExecutionProvider.TensorrtExecutionProvider,
        ExecutionProvider.NvTensorRTRTXExecutionProvider,
        ExecutionProvider.OpenVINOExecutionProvider,
        ExecutionProvider.JsExecutionProvider,
    },
    "npu": {
        ExecutionProvider.DmlExecutionProvider,
        ExecutionProvider.QNNExecutionProvider,
        ExecutionProvider.VitisAIExecutionProvider,
        ExecutionProvider.OpenVINOExecutionProvider,
    },
}
