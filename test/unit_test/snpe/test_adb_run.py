# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
import shutil
from pathlib import Path
from subprocess import CompletedProcess
from unittest.mock import patch

import pytest

from olive.common.constants import OS
from olive.platform_sdk.qualcomm.runner import SDKRunner
from olive.platform_sdk.qualcomm.snpe.utils.adb import run_adb_command

# pylint: disable=redefined-outer-name, unused-variable


@pytest.fixture
def android_target():
    return "emulator-5554"


@patch("subprocess.run")
def test_run_adb_command(mock_run_subprocess, android_target):
    ret_val = CompletedProcess(None, returncode=0, stdout=b"stdout", stderr=b"stderr")
    mock_run_subprocess.return_value = ret_val
    stdout, stderr = run_adb_command("version", android_target)
    mock_run_subprocess.assert_called_once_with(
        f"adb -s {android_target} version".split(), capture_output=True, env=None, cwd=None, check=False
    )
    assert stdout == "stdout"
    assert stderr == "stderr"


def test_run_snpe_command():
    if platform.system() == OS.WINDOWS:
        os.environ["SNPE_ROOT"] = "C:\\snpe"
        target_arch = "x86_64-windows-msvc"
    else:
        os.environ["SNPE_ROOT"] = "/snpe"
        target_arch = "x86_64-linux-clang"

    with patch.object(Path, "exists") as mock_exists, patch.object(Path, "glob") as mock_glob, patch.object(
        Path, "open"
    ) as open_file, patch("subprocess.run") as mock_run_subprocess:
        mock_exists.return_value = True
        mock_glob.return_value = [Path("lib") / target_arch]
        open_file.return_value.__enter__.return_value.readline.return_value = "python"
        mock_run_subprocess.return_value = CompletedProcess(None, returncode=0, stdout=b"stdout", stderr=b"stderr")
        runner = SDKRunner(platform="SNPE")
        stdout, _ = runner.run(cmd="snpe-net-run --container xxxx")
        if platform.system() == OS.LINUX:
            env = {
                "LD_LIBRARY_PATH": "/snpe/lib/x86_64-linux-clang",
                "PATH": f"/snpe/bin/x86_64-linux-clang:/usr/bin:{os.environ['PATH']}",
                "PYTHONPATH": "/snpe/lib/python",
                "SDK_ROOT": "/snpe",
            }
            cmd = "snpe-net-run --container xxxx"
        else:
            env = {
                "PATH": f"C:\\snpe\\bin\\x86_64-windows-msvc;C:\\snpe\\lib\\x86_64-windows-msvc;{os.environ['PATH']}",
                "SDK_ROOT": "C:\\snpe",
                "PYTHONPATH": "C:\\snpe\\lib\\python",
            }
            os_env = os.environ.copy()
            os_env.update(env)
            env = os_env
            python_cmd_path = shutil.which("python", path=env["PATH"])
            cmd = f"{python_cmd_path} C:\\snpe\\bin\\x86_64-windows-msvc\\snpe-net-run.exe --container xxxx"

        mock_run_subprocess.assert_called_once_with(
            cmd.split(),
            capture_output=True,
            env=env,
            cwd=None,
            check=True,
        )
        assert stdout.strip() == "stdout"
