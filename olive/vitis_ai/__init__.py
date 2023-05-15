#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#
from onnxruntime.quantization.calibrate import CalibrationDataReader
from onnxruntime.quantization.quant_utils import QuantFormat, QuantType
from olive.vitis_ai.quant_utils import PowerOfTwoMethod
from olive.vitis_ai.qdq_quantizer import VitisQuantizer
from olive.vitis_ai.quantize import quantize_static
