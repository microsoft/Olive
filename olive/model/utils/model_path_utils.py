# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def resolve_model_path(file_or_dir_path: str, model_filename: str) -> str:
    """Get the model full path.

    The engine provides output paths to Olive passes that do not contain model extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    resolve_onnx_path("c:/foo/bar.onnx", "model.onnx") -> c:/foo/bar.onnx
    resolve_onnx_path("c:/foo/bar.so", "model.so") -> c:/foo/bar.so

    resolve_onnx_path("c:/foo/bar",  "model.onnx") -> c:/foo/bar/model.onnx
    resolve_onnx_path("c:/foo/bar",  "model.so") -> c:/foo/bar/model.so
    """
    suffix = model_filename.split(".")[-1]
    if not suffix:
        raise ValueError(f"Model filename {model_filename} does not have a valid extension")

    path = Path(file_or_dir_path).resolve()
    if path.suffix != suffix:
        path = path / model_filename
        path.parent.mkdir(parents=True, exist_ok=True)
    return str(path)


def resolve_onnx_path(file_or_dir_path: str, model_filename: str = "model.onnx") -> str:
    """Get the model full path.

    The engine provides output paths to ONNX passes that do not contain .onnx extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    resolve_onnx_path("c:/foo/bar.onnx") -> c:/foo/bar.onnx

    resolve_onnx_path("c:/foo/bar") -> c:/foo/bar/model.onnx
    """
    return resolve_model_path(file_or_dir_path, model_filename)
