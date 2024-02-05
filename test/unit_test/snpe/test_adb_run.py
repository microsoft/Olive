# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from olive.platform_sdk.qualcomm.runner import SDKRunner
from olive.platform_sdk.qualcomm.snpe.utils.adb import run_adb_command

# pylint: disable=redefined-outer-name, unused-variable


@pytest.fixture()
def android_target():
    return "emulator-5554"


@patch("shutil.which")
@patch("subprocess.run")
def test_run_adb_command(mock_run_subprocess, mock_which, android_target):
    ret_val = CompletedProcess(None, returncode=0, stdout=b"stdout", stderr=b"stderr")
    mock_run_subprocess.return_value = ret_val
    mock_which.side_effect = lambda x, path: x
    stdout, stderr = run_adb_command("version", android_target)
    mock_run_subprocess.assert_called_once_with(
        f"adb -s {android_target} version".split(), capture_output=True, env=None, cwd=None, check=False
    )
    assert stdout == "stdout"
    assert stderr == "stderr"


def test_run_snpe_command():
    if platform.system() == "Windows":
        os.environ["SNPE_ROOT"] = "C:\\snpe"
        target_arch = "x86_64-windows-msvc"
    else:
        os.environ["SNPE_ROOT"] = "/snpe"
        target_arch = "x86_64-linux-clang"

    with patch.object(Path, "exists") as mock_exists, patch.object(Path, "glob") as mock_glob, patch(
        "shutil.which"
    ) as mock_witch, patch("subprocess.run") as mock_run_subprocess:
        mock_exists.return_value = True
        mock_glob.return_value = [Path("lib") / target_arch]
        mock_witch.side_effect = lambda x, path: x
        mock_run_subprocess.return_value = CompletedProcess(None, returncode=0, stdout=b"stdout", stderr=b"stderr")
        runner = SDKRunner(platform="SNPE")
        stdout, _ = runner.run(cmd="snpe-net-run --container xxxx")
        if platform.system() == "Linux":
            env = {
                "LD_LIBRARY_PATH": "/snpe/lib/x86_64-linux-clang",
                "PATH": "/snpe/bin/x86_64-linux-clang:/usr/bin",
                "PYTHONPATH": "/snpe/lib/python",
                "SDK_ROOT": "/snpe",
            }
        else:
            env = {
                "PATH": "C:\\snpe\\bin\\x86_64-windows-msvc;C:\\snpe\\lib\\x86_64-windows-msvc",
                "SDK_ROOT": "C:\\snpe",
                "PYTHONPATH": "C:\\snpe\\lib\\python",
            }

        mock_run_subprocess.assert_called_once_with(
            "snpe-net-run --container xxxx".split(),
            capture_output=True,
            env=env,
            cwd=None,
            check=True,
        )
        assert stdout.strip() == "stdout"
