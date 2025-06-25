# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.utils import all_files, copy_dir


@pytest.fixture(name="create_dir")
def create_dir_fixture(tmp_path):
    src_dir = tmp_path / "src_dir"
    src_dir.mkdir(parents=True, exist_ok=True)
    sub_dirs = ["sub_dir1"]
    files = ["file1.ext1", "file2.ext2", "sub_dir1/file3.ext1", "sub_dir1/file4.ext2"]
    for sub_dir in sub_dirs:
        (src_dir / sub_dir).mkdir(parents=True, exist_ok=True)
    for file in files:
        (src_dir / file).touch()
    return src_dir


@pytest.mark.parametrize(
    ("ignore", "expected_files"),
    [
        (None, ["file1.ext1", "file2.ext2", "sub_dir1/file3.ext1", "sub_dir1/file4.ext2"]),
        (shutil.ignore_patterns("*.ext1"), ["file2.ext2", "sub_dir1/file4.ext2"]),
        (shutil.ignore_patterns("*.ext2"), ["file1.ext1", "sub_dir1/file3.ext1"]),
        (shutil.ignore_patterns("*.ext*"), []),
        (shutil.ignore_patterns("*.ext1", "*.ext2"), []),
        (shutil.ignore_patterns("sub_dir1"), ["file1.ext1", "file2.ext2"]),
        (shutil.ignore_patterns("sub_dir1", "*.ext1"), ["file2.ext2"]),
        (shutil.ignore_patterns("sub_dir1", "*.ext2"), ["file1.ext1"]),
        (shutil.ignore_patterns("sub_dir1", "*.ext*"), []),
    ],
)
def test_all_files(create_dir, ignore, expected_files):
    actual_files = {file.relative_to(create_dir) for file in all_files(create_dir, ignore=ignore)}
    expected_files = {Path(file) for file in expected_files}
    assert actual_files == expected_files


def test_copy_dir(create_dir, tmp_path):
    # setup
    src_path = create_dir
    dest_path = tmp_path / "dest_dir"

    # test
    copy_dir(src_path, dest_path)

    # assert
    for file in src_path.glob("**/*"):
        assert (dest_path / file.relative_to(src_path)).exists()


def test_copy_dir_raise_file_exists_error(create_dir, tmp_path):
    # setup
    src_path = create_dir
    dest_path = tmp_path / "dest_dir"
    dest_path.mkdir(parents=True, exist_ok=True)
    (dest_path / "file1.ext1").touch()

    # test
    with pytest.raises(FileExistsError):
        copy_dir(src_path, dest_path)


@patch("shutil.copytree", side_effect=shutil.Error("Test Error"))
def test_copy_dir_raise_from_shutil_error(_, create_dir, tmp_path):
    # setup
    src_path = create_dir
    dest_path = tmp_path / "dest_dir"

    # test
    with pytest.raises(RuntimeError, match="Failed to copy *"):
        copy_dir(src_path, dest_path)


@patch("shutil.copytree", side_effect=shutil.Error("Test Error"))
def test_copy_dir_ignore_shutil_error(_, tmp_path, caplog):
    # setup
    src_path = tmp_path / "src_dir"
    src_path.mkdir(parents=True, exist_ok=True)
    (src_path / "file1.ext1").touch()

    dest_path = tmp_path / "dest_dir"
    dest_path.mkdir(parents=True, exist_ok=True)
    shutil.copy(src_path / "file1.ext1", dest_path / "file1.ext1")

    # The olive logger has propagate=False, so we need to temporarily enable it
    # to allow caplog to capture the logs
    olive_logger = logging.getLogger("olive")
    original_propagate = olive_logger.propagate
    olive_logger.propagate = True

    try:
        with caplog.at_level(logging.WARNING, logger="olive"):
            copy_dir(src_path, dest_path, ignore_errors=True)

        # assert
        assert "Assuming all files were copied successfully and continuing." in caplog.text
    finally:
        # Restore original propagate setting
        olive_logger.propagate = original_propagate
