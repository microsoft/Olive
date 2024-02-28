# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from pathlib import Path


def normalize_path_suffix(file_or_dir_path: str, filename_with_suffix: str) -> str:
    """Get the file full path.

    The engine provides output paths to Olive passes that do not contain file extension
    (these paths are generally locations in the cache). This function will convert such
    paths to absolute file paths and also ensure the parent directories exist.
    If the input path is already an ONNX file it is simply returned. Examples:

    normalize_path_suffix("c:/foo/bar.onnx", "model.onnx") -> c:/foo/bar.onnx
    normalize_path_suffix("c:/foo/bar.so", "model.so") -> c:/foo/bar.so

    normalize_path_suffix("c:/foo/bar",  "model.onnx") -> c:/foo/bar/model.onnx
    normalize_path_suffix("c:/foo/bar",  "model.so") -> c:/foo/bar/model.so
    """
    if not (suffix := Path(filename_with_suffix).suffix):
        raise ValueError(f"{filename_with_suffix} does not have a valid extension")

    path = Path(file_or_dir_path).resolve()
    if path.is_file() and path.suffix != suffix:
        raise ValueError(f"{path} if a file but does not have a valid extension {suffix}")
    if path.suffix != suffix:
        path.mkdir(parents=True, exist_ok=True)
        path = path / filename_with_suffix
    return str(path)
