# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.utils import StrEnumBase


class Framework(StrEnumBase):
    """Framework of the model."""

    ONNX = "ONNX"
    PYTORCH = "PyTorch"
    QNN = "QNN"
    TENSORFLOW = "TensorFlow"
    SNPE = "SNPE"
    OPENVINO = "OpenVINO"


class ModelFileFormat(StrEnumBase):
    """Given a framework, there might be 1 or more on-disk model file format(s), model save/Load logic may differ."""

    ONNX = "ONNX"
    PYTORCH_ENTIRE_MODEL = "PyTorch.EntireModel"
    PYTORCH_STATE_DICT = "PyTorch.StateDict"
    PYTORCH_TORCH_SCRIPT = "PyTorch.TorchScript"
    PYTORCH_SLICE_GPT_MODEL = "PyTorch.SliceGPT"
    TENSORFLOW_PROTOBUF = "TensorFlow.Protobuf"
    TENSORFLOW_SAVED_MODEL = "TensorFlow.SavedModel"
    SNPE_DLC = "SNPE.DLC"
    QNN_CPP = "QNN.CPP"
    QNN_LIB = "QNN.LIB"
    QNN_SERIALIZED_BIN = "QNN.SERIALIZED.BIN"
    OPENVINO_IR = "OpenVINO.IR"
    COMPOSITE_MODEL = "Composite"


class Precision(StrEnumBase):
    INT4 = "int4"
    INT8 = "int8"
    INT16 = "int16"
    INT32 = "int32"
    UINT4 = "uint4"
    UINT8 = "uint8"
    UINT16 = "uint16"
    UINT32 = "uint32"
    FP4 = "fp4"
    FP8 = "fp8"
    FP16 = "fp16"
    FP32 = "fp32"
    NF4 = "nf4"


class QuantAlgorithm(StrEnumBase):
    AWQ = "awq"
    GPTQ = "gptq"
    HQQ = "hqq"
    RTN = "rtn"
    SPINQUANT = "spinquant"
    QUAROT = "quarot"


class QuantEncoding(StrEnumBase):
    QDQ = "qdq"
    QOP = "qop"


class DatasetRequirement(StrEnumBase):
    REQUIRED = "dataset_required"
    OPTIONAL = "dataset_optional"
    NOT_REQUIRED = "dataset_not_required"
