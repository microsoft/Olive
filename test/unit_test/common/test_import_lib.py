# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import pytest

from olive.common.import_lib import import_user_module


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_user_script_is_file(mock_importlib_util, mock_sys_path):
    """Test import_user_module when user_script is a file in script_dir."""
    # setup
    user_script = "user_script_a.py"
    script_dir = "script_dir_a"

    Path(script_dir).mkdir(parents=True, exist_ok=True)
    script_dir_path = Path(script_dir).resolve()

    # put user_script in script_dir
    user_script_path = script_dir_path / user_script
    with open(user_script_path, "w") as _:
        pass

    # mock
    mock_spec = MagicMock()
    mock_importlib_util.find_spec.return_value = mock_spec
    expected_res = MagicMock()
    mock_importlib_util.module_from_spec.return_value = expected_res

    # execute
    actual_res = import_user_module(user_script, script_dir)

    # assert
    assert actual_res == expected_res
    # script_dir will be added to sys.path
    mock_sys_path.append.assert_called_once_with(str(script_dir_path))
    # mock_importlib_util can find the user_script
    mock_importlib_util.find_spec.assert_called_once_with("user_script_a")
    mock_importlib_util.spec_from_file_location.assert_not_called()
    mock_importlib_util.module_from_spec.assert_called_once_with(mock_spec)
    mock_spec.loader.exec_module.assert_called_once_with(expected_res)

    # cleanup
    if os.path.exists(script_dir_path):
        shutil.rmtree(script_dir_path)
    if os.path.exists(user_script_path):
        os.remove(user_script_path)


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_user_script_is_dir(mock_importlib_util, mock_sys_path):
    """Test import_user_module when.

    - script_dir is None
    - user_script is a dir
    """
    # setup
    user_script = "user_script_b"

    Path(user_script).mkdir(parents=True, exist_ok=True)
    user_script_path = Path(user_script).resolve()
    with open(user_script_path / "__init__.py", "w") as _:
        pass

    mock_spec = MagicMock()
    mock_importlib_util.find_spec.return_value = None
    mock_importlib_util.spec_from_file_location.return_value = mock_spec
    expected_res = MagicMock()
    mock_importlib_util.module_from_spec.return_value = expected_res

    # execute
    actual_res = import_user_module(user_script, script_dir=None)

    # assert
    assert actual_res == expected_res
    mock_sys_path.append.assert_not_called()
    mock_importlib_util.find_spec.assert_called_once_with("user_script_b")
    mock_importlib_util.spec_from_file_location.assert_called_once_with(
        "user_script_b", (user_script_path / "__init__.py").resolve()
    )
    mock_importlib_util.module_from_spec.assert_called_once_with(mock_spec)
    mock_spec.loader.exec_module.assert_called_once_with(expected_res)

    # cleanup
    if os.path.exists(user_script_path):
        shutil.rmtree(user_script_path)


def test_import_user_module_script_dir_exception():
    # setup
    user_script = "user_script_c"
    script_dir = "script_dir_c"

    # execute
    with pytest.raises(ValueError) as errinfo:  # noqa: PT011
        import_user_module(user_script, script_dir)

    # assert
    script_dir_path = Path(script_dir).resolve()
    assert str(errinfo.value) == f"{script_dir_path} doesn't exist"


def test_import_user_module_user_script_exception():
    # setup
    user_script = "user_script_d"

    # execute
    with pytest.raises(ValueError) as errinfo:  # noqa: PT011
        import_user_module(user_script, script_dir=None)

    # assert
    user_script_path = Path(user_script).resolve()
    assert str(errinfo.value) == f"{user_script_path} doesn't exist"


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_script_dir_none_and_user_script_exists(mock_importlib_util, mock_sys_path):
    """Test import_user_module when.

    1. script_dir is None
    2. user_script is not in any dir in sys.path
    3. user_script exists
    """
    with TemporaryDirectory(prefix="not_in_sys_path_dir") as temp_dir:
        # setup
        user_script = "user_script_e.py"
        user_script_path = Path(temp_dir) / user_script
        with open(user_script_path, "w") as _:
            pass

        # mock
        mock_spec = MagicMock()
        mock_importlib_util.find_spec.return_value = None
        mock_importlib_util.spec_from_file_location.return_value = mock_spec
        expected_module = MagicMock()
        mock_importlib_util.module_from_spec.return_value = expected_module

        # execute
        actual_res = import_user_module(user_script_path, script_dir=None)

        # assert
        assert actual_res == expected_module
        mock_sys_path.append.assert_not_called()
        mock_importlib_util.find_spec.assert_called_once_with("user_script_e")
        mock_importlib_util.spec_from_file_location.assert_called_once_with("user_script_e", user_script_path.resolve())
        mock_importlib_util.module_from_spec.assert_called_once_with(mock_spec)
        mock_spec.loader.exec_module.assert_called_once_with(expected_module)


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_script_dir_none_and_user_script_not_exists(mock_importlib_util, mock_sys_path):
    """Test import_user_module when.

    1. script_dir is None
    2. user_script is not in any dir in sys.path
    3. user_script does not exist
    """
    # setup
    user_script = "nonexistent_script.py"

    # mock
    mock_importlib_util.find_spec.return_value = None

    # execute
    with pytest.raises(ValueError) as errinfo:  # noqa: PT011
        import_user_module(user_script, script_dir=None)

    # assert
    assert str(errinfo.value) == f"{Path(user_script).resolve()} doesn't exist"
    mock_sys_path.append.assert_not_called()
    mock_importlib_util.find_spec.assert_called_once_with("nonexistent_script")
    mock_importlib_util.spec_from_file_location.assert_not_called()
    mock_importlib_util.module_from_spec.assert_not_called()


def test_import_user_module_user_script_in_sys_path():
    """Test import_user_module with the following conditions.

    1. user_script is in a directory already in sys.path.
    2. script_dir is None.
    3. find_spec is used, and spec_from_file_location is not called.
    """
    with TemporaryDirectory(prefix="temp_sys_path_dir") as temp_dir:
        # setup
        temp_dir_path = Path(temp_dir).resolve()
        user_script = "user_script_f.py"
        user_script_path = temp_dir_path / user_script
        with open(user_script_path, "w") as _:
            pass

        # add temp_dir to sys.path
        import sys

        sys.path.insert(0, str(temp_dir_path))

        try:
            # execute
            actual_res = import_user_module(user_script, script_dir=None)

            # assert
            assert actual_res.__file__ == str(user_script_path)
        finally:
            sys.path.remove(str(temp_dir_path))
