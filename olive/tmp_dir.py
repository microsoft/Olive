# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import tempfile
from pathlib import Path

logger = logging.getLogger(__name__)


def get_tmp_dir_root(mkdir: bool = True) -> str:
    """Get the root directory for temporary files.

    :param mkdir: whether to create the directory if it does not exist
    :return: the root directory for temporary files
    """
    tmp_dir_root = os.environ.get("OLIVE_TMP_DIR_ROOT", None)
    return _resolve_and_create_dir(tmp_dir_root, mkdir) if tmp_dir_root is not None else None


def set_tmp_dir_root(root: str) -> str:
    """Set the root directory for temporary files.

    :param root: the root directory for temporary files
    :return: the resolved path to the root directory
    """
    # resolve and create the directory if it does not exist
    root = _resolve_and_create_dir(root)

    # set the environment variable
    if get_tmp_dir_root() is not None:
        logger.debug(f"Changing OLIVE_TMP_DIR_ROOT from {get_tmp_dir_root(mkdir=False)} to {root}")
    else:
        logger.debug(f"Setting OLIVE_TMP_DIR_ROOT to {root}")
    os.environ["OLIVE_TMP_DIR_ROOT"] = root

    return root


def reset_tmp_dir_root():
    """Reset the root directory for temporary files."""
    if get_tmp_dir_root() is not None:
        logger.debug(f"Resetting OLIVE_TMP_DIR_ROOT from {get_tmp_dir_root(mkdir=False)} to None")
    else:
        logger.debug("OLIVE_TMP_DIR_ROOT is already None")

    os.environ.pop("OLIVE_TMP_DIR_ROOT", None)


def _resolve_and_create_dir(dir_path: str, mkdir: bool = True) -> str:
    """Resolve the path to avoid ambiguity due to relative paths, and create the directory if it does not exist.

    :param dir_path: the path to the directory
    :param mkdir: whether to create the directory if it does not exist
    :return: the resolved path to the directory
    """
    # resolve the path to avoid ambiguity due to relative paths
    dir_path = Path(dir_path).resolve()
    # create the directory if it does not exist
    if mkdir:
        dir_path.mkdir(parents=True, exist_ok=True)

    return str(dir_path)


def get_temporary_directory(suffix: str = None, prefix: str = "olive_tmp_"):
    """Get a tempfile.TemporaryDirectory object created under the OLIVE_TMP_DIR_ROOT directory.

    If OLIVE_TMP_DIR_ROOT is not set, the default temporary directory will be used.

    :param suffix: the suffix of the temporary directory name
    :param prefix: the prefix of the temporary directory name
    :return: a tempfile.TemporaryDirectory object
    """
    return tempfile.TemporaryDirectory(suffix=suffix, prefix=prefix, dir=get_tmp_dir_root())


def get_named_temporary_file(**kwargs):
    """Create and return a temporary file. under the root directory for temporary files.

    If OLIVE_TMP_DIR_ROOT is not set, the default temporary directory will be used.

    :param kwargs: keyword arguments for tempfile.NamedTemporaryFile
    :return: a file-like object
    """
    if "dir" in kwargs:
        assert kwargs["dir"] is None, "dir cannot be specified in kwargs"
    kwargs["dir"] = get_tmp_dir_root()

    return tempfile.NamedTemporaryFile(**kwargs)
