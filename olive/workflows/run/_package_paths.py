# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Path checks shared by component-build preflight and ONNX package publication."""

import os
import stat
from pathlib import Path


def _is_link(path: Path) -> bool:
    if path.is_symlink() or bool(getattr(path, "is_junction", lambda: False)()):
        return True
    if os.name == "nt":
        try:
            return bool(path.lstat().st_file_attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)
        except FileNotFoundError:
            pass
    return False


def reject_links(path: Path) -> None:
    """Reject existing symlinks and junctions in a path or any of its parents."""
    for part in (path, *path.parents):
        if _is_link(part):
            raise ValueError(f"ONNX component packages cannot use a symlink or junction: {part}")


def validate_source_tree(source: Path) -> None:
    """Reject linked assets before copying a package, including linked directories."""
    reject_links(source)
    for path in source.rglob("*"):
        if _is_link(path):
            raise ValueError(f"ONNX component packages cannot contain a symlink or junction: {path}")


def destination_path(root: Path, relative: Path) -> Path:
    """Resolve a package destination without following links outside the output root."""
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError(f"Invalid ONNX package relative path: {relative}")
    reject_links(root)
    destination = root / relative
    reject_links(destination)
    resolved_root = root.resolve()
    if resolved_root not in destination.resolve().parents:
        raise ValueError(f"ONNX package destination {destination} escapes output root {root}")
    return destination


def confined_artifact_path(root: Path, path: Path) -> Path:
    """Keep a model asset inside its build output, without following links."""
    reject_links(root)
    reject_links(path)
    resolved_root = root.resolve()
    resolved_path = path.resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"ONNX artifact {path} escapes build output root {root}")
    if not resolved_path.exists():
        raise FileNotFoundError(f"ONNX artifact does not exist: {path}")
    return resolved_path


def confined_artifact_file(root: Path, path: Path) -> Path:
    """Validate an ONNX model or external file before opening or copying it."""
    resolved_path = confined_artifact_path(root, path)
    if not resolved_path.is_file():
        raise ValueError(f"ONNX artifact is not a file: {path}")
    return resolved_path
