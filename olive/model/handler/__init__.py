# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model.handler.base import OliveModelHandler
from olive.model.handler.composite import CompositeModelHandler, CompositePyTorchModelHandler
from olive.model.handler.hf import HfModelHandler
from olive.model.handler.onnx import DistributedOnnxModelHandler, ONNXModelHandler
from olive.model.handler.openvino import OpenVINOModelHandler
from olive.model.handler.pytorch import DistributedPyTorchModelHandler, PyTorchModelHandler
from olive.model.handler.pytorch2 import DistributedPyTorchModelHandler2, PyTorchModelHandler2
from olive.model.handler.qnn import QNNModelHandler
from olive.model.handler.snpe import SNPEModelHandler
from olive.model.handler.tensorflow import TensorFlowModelHandler

__all__ = [
    "OliveModelHandler",
    "CompositeModelHandler",
    "CompositePyTorchModelHandler",
    "DistributedOnnxModelHandler",
    "DistributedPyTorchModelHandler2",
    "HfModelHandler",
    "ONNXModelHandler",
    "OpenVINOModelHandler",
    "DistributedPyTorchModelHandler",
    "PyTorchModelHandler",
    "PyTorchModelHandler2",
    "QNNModelHandler",
    "SNPEModelHandler",
    "TensorFlowModelHandler",
]
