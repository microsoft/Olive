#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType

from olive.passes.onnx.vitis_ai.quant_utils import PowerOfTwoMethod
from olive.passes.onnx.vitis_ai.quantize import quantize_static

__all__ = [
    "CalibrationDataReader",
    "PowerOfTwoMethod",
    "QuantFormat",
    "QuantType",
    "quantize_static",
]
