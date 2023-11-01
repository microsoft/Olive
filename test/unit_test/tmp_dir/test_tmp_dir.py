# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import shutil
from pathlib import Path

import pytest

from olive.tmp_dir import (
    get_named_temporary_file,
    get_temporary_directory,
    get_tmp_dir_root,
    reset_tmp_dir_root,
    set_tmp_dir_root,
)


# Note: Only use 'olive_tmp_dir' as the root directory for test cases!
# Otherwise, the root dir won't be cleaned up automatically after the test cases are finished.
class TestTmpDir:
    @pytest.fixture(autouse=True)
    def setup(self):
        reset_tmp_dir_root()

        yield

        reset_tmp_dir_root()
        shutil.rmtree("olive_tmp_dir", ignore_errors=True)

    @pytest.fixture
    def olive_tmp_dir_root(self):
        root = "olive_tmp_dir"
        set_tmp_dir_root(root)
        return Path(root).resolve()

    @pytest.mark.parametrize("root", [None, "", ".", "olive_tmp_dir"])
    def test_get_tmp_dir_root(self, root: str):
        if root is not None:
            os.environ["OLIVE_TMP_DIR_ROOT"] = root

        expected_root = str(Path(root).resolve()) if root is not None else None
        assert get_tmp_dir_root() == expected_root

    @pytest.mark.parametrize(
        "root,fail",
        [(None, True), (1, True), ("", False), (".", False), ("olive_tmp_dir", False), (Path("olive_tmp_dir"), False)],
    )
    def test_set_tmp_dir_root(self, root: str, fail: bool):
        if fail:
            with pytest.raises(TypeError):
                set_tmp_dir_root(root)
        else:
            set_tmp_dir_root(root)
            assert get_tmp_dir_root() == str(Path(root).resolve())

    def test_reset_tmp_dir_root(self):
        set_tmp_dir_root("olive_tmp_dir")
        reset_tmp_dir_root()
        assert get_tmp_dir_root() is None

    def _check_tmp_path(self, tmp_path_str: str, root_path: Path, is_dir: bool = True):
        assert Path(tmp_path_str).resolve().parent == root_path
        assert Path(tmp_path_str).resolve().name.startswith("test_")
        assert Path(tmp_path_str).is_dir() if is_dir else Path(tmp_path_str).is_file()

    def test_get_temporary_directory_object(self, olive_tmp_dir_root):
        temp_dir = get_temporary_directory(prefix="test_")
        self._check_tmp_path(temp_dir.name, olive_tmp_dir_root)

        temp_dir.cleanup()
        assert not Path(temp_dir.name).exists()

    def test_get_temporary_directory_context_manager(self, olive_tmp_dir_root: Path):
        with get_temporary_directory(prefix="test_") as temp_dir:
            self._check_tmp_path(temp_dir, olive_tmp_dir_root)

        assert not Path(temp_dir).exists()

    def test_get_named_temporary_file_object(self, olive_tmp_dir_root: Path):
        temp_file = get_named_temporary_file(prefix="test_")
        self._check_tmp_path(temp_file.name, olive_tmp_dir_root, is_dir=False)

        temp_file.close()
        assert not Path(temp_file.name).exists()

    def test_get_named_temporary_file_context_manager(self, olive_tmp_dir_root: Path):
        temp_file_name = None
        with get_named_temporary_file(prefix="test_") as temp_file:
            temp_file_name = temp_file.name
            self._check_tmp_path(temp_file.name, olive_tmp_dir_root, is_dir=False)

        assert not Path(temp_file_name).exists()
