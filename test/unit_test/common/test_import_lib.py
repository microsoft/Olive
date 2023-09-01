# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import shutil
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.common.import_lib import import_user_module


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_user_script_is_file(mock_importlib_util, mock_sys_path):
    # setup
    user_script = "user_script.py"
    script_dir = "script_dir"

    Path(script_dir).mkdir(parents=True, exist_ok=True)

    open(user_script, "w")
    mock_spec = MagicMock()
    mock_importlib_util.spec_from_file_location.return_value = mock_spec
    expected_res = MagicMock()
    mock_importlib_util.module_from_spec.return_value = expected_res

    # execute
    actual_res = import_user_module(user_script, script_dir)

    # assert
    script_dir_path = Path(script_dir).resolve()
    mock_sys_path.append.assert_called_once_with(str(script_dir_path))
    assert actual_res == expected_res

    user_script_path = Path(user_script).resolve()
    mock_importlib_util.spec_from_file_location.assert_called_once_with("user_script", user_script_path)
    mock_importlib_util.module_from_spec.assert_called_once_with(mock_spec)

    # cleanup
    if os.path.exists(script_dir_path):
        shutil.rmtree(script_dir_path)
    if os.path.exists(user_script_path):
        os.remove(user_script_path)


@patch("olive.common.import_lib.sys.path")
@patch("olive.common.import_lib.importlib.util")
def test_import_user_module_user_script_is_dir(mock_importlib_util, mock_sys_path):
    # setup
    user_script = "user_script"
    script_dir = "script_dir"

    Path(script_dir).mkdir(parents=True, exist_ok=True)
    Path(user_script).mkdir(parents=True, exist_ok=True)

    mock_spec = MagicMock()
    mock_importlib_util.spec_from_file_location.return_value = mock_spec
    expected_res = MagicMock()
    mock_importlib_util.module_from_spec.return_value = expected_res

    # execute
    actual_res = import_user_module(user_script, script_dir)

    # assert
    script_dir_path = Path(script_dir).resolve()
    mock_sys_path.append.assert_called_once_with(str(script_dir_path))
    assert actual_res == expected_res

    user_script_path = Path(user_script).resolve()
    user_script_path_init = user_script_path / "__init__.py"
    mock_importlib_util.spec_from_file_location.assert_called_once_with("user_script", user_script_path_init)
    mock_importlib_util.module_from_spec.assert_called_once_with(mock_spec)

    # cleanup
    if os.path.exists(script_dir_path):
        shutil.rmtree(script_dir_path)
    if os.path.exists(user_script_path):
        shutil.rmtree(user_script_path)


def test_import_user_module_script_dir_exception():
    # setup
    user_script = "user_script"
    script_dir = "script_dir"

    # execute
    with pytest.raises(ValueError) as errinfo:
        import_user_module(user_script, script_dir)

    # assert
    script_dir_path = Path(script_dir).resolve()
    assert str(errinfo.value) == f"{script_dir_path} doesn't exist"


def test_import_user_module_user_script_exception():
    # setup
    user_script = "user_script"

    # execute
    with pytest.raises(ValueError) as errinfo:
        import_user_module(user_script)

    # assert
    user_script_path = Path(user_script).resolve()
    assert str(errinfo.value) == f"{user_script_path} doesn't exist"
