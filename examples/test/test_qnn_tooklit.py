# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path

import pytest
from utils import check_output, download_azure_blob

from olive.common.utils import retry_func, run_subprocess


class TestQnnToolkit:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Download the qnn sdk."""
        blob, download_path = "", ""
        if platform.system() == "Windows":
            blob, download_path = "qnn_sdk_windows.zip", "qnn_sdk_windows.zip"
        elif platform.system() == "Linux":
            blob, download_path = "qnn_sdk_linux.zip", "qnn_sdk_linux.zip"

        download_azure_blob(
            container="olivetest",
            blob=blob,
            download_path=download_path,
        )
        target_path = tmp_path / "qnn_sdk"
        target_path.mkdir(parents=True, exist_ok=True)
        if platform.system() == "Windows":
            cmd = f"powershell Expand-Archive -Path {download_path} -DestinationPath {str(target_path)}"
            run_subprocess(cmd=cmd, check=True)
        elif platform.system() == "Linux":
            run_subprocess(cmd=f"unzip {download_path} -d {str(target_path)}", check=True)

        os.environ["QNN_SDK_ROOT"] = str(target_path / "opt" / "qcom" / "aistack")

    def _setup_resource(self, use_olive_env):
        """Setups any state specific to the execution of the given module."""
        cur_dir = Path(__file__).resolve().parent.parent
        example_dir = cur_dir / "mobilenet"
        os.chdir(example_dir)

        if use_olive_env:
            os.environ["USE_OLIVE_ENV"] = "1"
            # prepare model and data
            # retry since it fails randomly
            run_subprocess(cmd="python -m olive.platform_sdk.qualcomm.configure --py_version 3.8 --sdk qnn", check=True)
            # install dependencies
            python_cmd = ""
            if platform.system() == "Windows":
                python_cmd = str(Path(os.environ["QNN_SDK_ROOT"]) / "olive-pyenv" / "python.exe")
            elif platform.system() == "Linux":
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
            if platform.system() == "Linux":
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
