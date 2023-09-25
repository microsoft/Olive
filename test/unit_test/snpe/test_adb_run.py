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

from olive.snpe.utils.adb import run_adb_command
from olive.snpe.utils.local import run_snpe_command


@pytest.fixture
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
        f"adb -s {android_target} version".split(), capture_output=True, env=None, cwd=None
    )
    assert stdout == "stdout"
    assert stderr == "stderr"


def test_run_snpe_command():
    os.environ["SNPE_ROOT"] = "C:\\snpe"
    with patch.object(Path, "exists") as mock_exists, patch.object(Path, "glob") as mock_glob, patch(
        "shutil.which"
    ) as mock_witch, patch("subprocess.run") as mock_run_subprocess:
        mock_exists.return_value = True
        mock_glob.return_value = [Path("lib") / "lib/x86_64-windows-vc19"]
        mock_witch.side_effect = lambda x, path: x
        mock_run_subprocess.return_value = CompletedProcess(None, returncode=0, stdout=b"stdout", stderr=b"stderr")
        stdout, stderr = run_snpe_command("snpe-net-run --container xxxx")
        if platform.system() == "Linux":
            env = {
                "LD_LIBRARY_PATH": "C:\\snpe/lib/x86_64-linux-clang",
                "PATH": "C:\\snpe/bin/x86_64-linux-clang:/usr/bin",
            }
        else:
            env = {"PATH": "C:\\snpe\\bin\\x86_64-windows-vc19;C:\\snpe\\lib\\x86_64-windows-vc19"}

        mock_run_subprocess.assert_called_once_with(
            "snpe-net-run --container xxxx".split(),
            capture_output=True,
            env=env,
            cwd=None,
        )
        assert stdout.strip() == "stdout"
