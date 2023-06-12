# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization

__all__ = [
    "OpenVINOConversion",
    "OpenVINOQuantization",
]
