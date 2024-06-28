# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path

import pytest
from utils import check_output, download_azure_blob

from olive.common.constants import OS
from olive.common.utils import retry_func, run_subprocess
from olive.logging import set_verbosity_debug

set_verbosity_debug()


class TestQnnToolkit:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Download the qnn sdk."""
        if platform.system() == OS.WINDOWS:
            blob, download_path = "qnn_sdk_windows.zip", "qnn_sdk_windows.zip"
            conda_installer_blob, conda_installer_path = (
                "conda-installers/Miniconda3-latest-Windows-x86_64.exe",
                tmp_path / "conda_installer.exe",
            )
        elif platform.system() == OS.LINUX:
            blob, download_path = "qnn_sdk_linux.zip", "qnn_sdk_linux.zip"
            conda_installer_blob, conda_installer_path = (
                "conda-installers/Miniconda3-latest-Linux-x86_64.sh",
                tmp_path / "conda_installer.sh",
            )
        else:
            raise NotImplementedError(f"Unsupported platform: {platform.system()}")

        download_azure_blob(
            container="olivetest",
            blob=blob,
            download_path=download_path,
        )
        target_path = tmp_path / "qnn_sdk"
        target_path.mkdir(parents=True, exist_ok=True)
        if platform.system() == OS.WINDOWS:
            cmd = f"powershell Expand-Archive -Path {download_path} -DestinationPath {str(target_path)}"
            run_subprocess(cmd=cmd, check=True)
        elif platform.system() == OS.LINUX:
            run_subprocess(cmd=f"unzip {download_path} -d {str(target_path)}", check=True)

        os.environ["QNN_SDK_ROOT"] = str(target_path / "opt" / "qcom" / "aistack")

        download_azure_blob(
            container="olivetest",
            blob=conda_installer_blob,
            download_path=conda_installer_path,
        )
        os.environ["CONDA_INSTALLER"] = str(conda_installer_path)

    def _setup_resource(self, use_olive_env):
        """Setups any state specific to the execution of the given module."""
        cur_dir = Path(__file__).resolve().parent.parent
        example_dir = cur_dir / "mobilenet"
        os.chdir(example_dir)

        if use_olive_env:
            os.environ["USE_OLIVE_ENV"] = "1"
            # retry since it fails randomly
            retry_func(
                run_subprocess,
                kwargs={"cmd": "olive configure-qualcomm-sdk --py_version 3.8 --sdk qnn", "check": True},
            )
            # install dependencies
            python_cmd = ""
            if platform.system() == OS.WINDOWS:
                python_cmd = str(Path(os.environ["QNN_SDK_ROOT"]) / "olive-pyenv" / "python.exe")
            elif platform.system() == OS.LINUX:
                python_cmd = str(Path(os.environ["QNN_SDK_ROOT"]) / "olive-pyenv" / "bin" / "python")
            install_cmd = [
                python_cmd,
                str(Path(os.environ["QNN_SDK_ROOT"]) / "bin" / "check-python-dependency"),
            ]
            run_subprocess(cmd=" ".join(install_cmd), check=True)
        else:
            os.environ["USE_OLIVE_ENV"] = "0"
            packages = ["tensorflow==2.10.1", "numpy==1.23.5"]
            retry_func(run_subprocess, kwargs={"cmd": f"python -m pip install {' '.join(packages)}", "check": True})
            os.environ["PYTHONPATH"] = str(Path(os.environ["QNN_SDK_ROOT"]) / "lib" / "python")
            if platform.system() == OS.LINUX:
                os.environ["PATH"] = (
                    str(Path(os.environ["QNN_SDK_ROOT"]) / "bin" / "x86_64-linux-clang")
                    + os.path.pathsep
                    + os.environ["PATH"]
                )
                os.environ["LD_LIBRARY_PATH"] = str(Path(os.environ["QNN_SDK_ROOT"]) / "lib" / "x86_64-linux-clang")
            else:
                os.environ["PATH"] = (
                    str(Path(os.environ["QNN_SDK_ROOT"]) / "bin" / "x86_64-windows-msvc")
                    + os.path.pathsep
                    + str(Path(os.environ["QNN_SDK_ROOT"]) / "lib" / "x86_64-windows-msvc")
                    + os.path.pathsep
                    + os.environ["PATH"]
                )
        retry_func(run_subprocess, kwargs={"cmd": "python download_files.py", "check": True})
        retry_func(run_subprocess, kwargs={"cmd": "python prepare_config.py --use_raw_qnn_sdk", "check": True})

    @pytest.mark.parametrize("use_olive_env", [True, False])
    def test_mobilenet_qnn(self, use_olive_env):
        from olive.workflows import run as olive_run

        self._setup_resource(use_olive_env)

        footprint = olive_run("raw_qnn_sdk_config.json")
        check_output(footprint)
