# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum


class Framework(str, Enum):
    """Framework of the model."""

    ONNX = "ONNX"
    PYTORCH = "PyTorch"
    TENSORFLOW = "TensorFlow"
    SNPE = "SNPE"
    OPENVINO = "OpenVINO"


class ModelFileFormat(str, Enum):
    """Given a framework, there might be 1 or more on-disk model file format(s), model save/Load logic may differ."""

    ONNX = "ONNX"
    PYTORCH_ENTIRE_MODEL = "PyTorch.EntireModel"
    PYTORCH_STATE_DICT = "PyTorch.StateDict"
    PYTORCH_TORCH_SCRIPT = "PyTorch.TorchScript"
    PYTORCH_MLFLOW_MODEL = "PyTorch.MLflow"
    TENSORFLOW_PROTOBUF = "TensorFlow.Protobuf"
    TENSORFLOW_SAVED_MODEL = "TensorFlow.SavedModel"
    SNPE_DLC = "SNPE.DLC"
    OPENVINO_IR = "OpenVINO.IR"
    OPTIMUM = "Optimum"
