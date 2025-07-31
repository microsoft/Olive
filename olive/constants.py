# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.common.config_utils import CaseInsensitiveEnum
from olive.common.utils import IntEnumBase, StrEnumBase

MSFT_DOMAIN = "com.microsoft"


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
    BF16 = "bf16"


class PrecisionBits(IntEnumBase):
    BITS4 = 4
    BITS8 = 8
    BITS16 = 16
    BITS32 = 32


class QuantAlgorithm(CaseInsensitiveEnum):
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


class OpType(StrEnumBase):
    """Enum for operator types."""

    Gather = "Gather"
    GatherBlockQuantized = "GatherBlockQuantized"
    MatMulNBits = "MatMulNBits"
    MatMul = "MatMul"
    QuickGelu = "QuickGelu"
    Sigmoid = "Sigmoid"
    Mul = "Mul"
    RotaryEmbedding = "RotaryEmbedding"
    Reshape = "Reshape"
    Slice = "Slice"
    Sub = "Sub"
    Add = "Add"
    Concat = "Concat"
    Div = "Div"
    Shape = "Shape"
    Constant = "Constant"


class AccuracyLevel(IntEnumBase):
    unset = 0
    fp32 = 1
    fp16 = 2
    bf16 = 3
    int8 = 4


def precision_bits_from_precision(p):
    mapping = {
        Precision.INT4: PrecisionBits.BITS4,
        Precision.INT8: PrecisionBits.BITS8,
        Precision.INT16: PrecisionBits.BITS16,
        Precision.INT32: PrecisionBits.BITS32,
        Precision.UINT4: PrecisionBits.BITS4,
        Precision.UINT8: PrecisionBits.BITS8,
        Precision.UINT16: PrecisionBits.BITS16,
        Precision.UINT32: PrecisionBits.BITS32,
    }
    return mapping.get(p)
