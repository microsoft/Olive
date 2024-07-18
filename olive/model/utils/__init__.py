# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model.utils.onnx_utils import resolve_onnx_path
from olive.model.utils.path_utils import normalize_path_suffix

__all__ = [
    "normalize_path_suffix",
    "resolve_onnx_path",
]
