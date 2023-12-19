# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.composite import CompositeModelHandler, CompositePyTorchModelHandler
from olive.model.handler.onnx import DistributedOnnxModelHandler, ONNXModelHandler
from olive.model.handler.openvino import OpenVINOModelHandler
from olive.model.handler.optimum import OptimumModelHandler
from olive.model.handler.pytorch import DistributedPyTorchModelHandler, PyTorchModelHandler
from olive.model.handler.snpe import SNPEModelHandler
from olive.model.handler.tensorflow import TensorFlowModelHandler

OliveModel = OliveModelHandler
CompositeModel = CompositeModelHandler
CompositePyTorchModel = CompositePyTorchModelHandler
DistributedOnnxModel = DistributedOnnxModelHandler
ONNXModel = ONNXModelHandler
OpenVINOModel = OpenVINOModelHandler
DistributedPyTorchModel = DistributedPyTorchModelHandler
PyTorchModel = PyTorchModelHandler
SNPEModel = SNPEModelHandler
TensorFlowModel = TensorFlowModelHandler


__all__ = [
    "OliveModelHandler",
    "CompositeModelHandler",
    "CompositePyTorchModelHandler",
    "DistributedOnnxModelHandler",
    "ONNXModelHandler",
    "OpenVINOModelHandler",
    "OptimumModelHandler",
    "DistributedPyTorchModelHandler",
    "PyTorchModelHandler",
    "SNPEModelHandler",
    "TensorFlowModelHandler",
    # alias
    "OliveModel",
    "CompositeModel",
    "CompositePyTorchModel",
    "DistributedOnnxModel",
    "ONNXModel",
    "OpenVINOModel",
    "DistributedPyTorchModel",
    "PyTorchModel",
    "SNPEModel",
    "TensorFlowModel",
]
