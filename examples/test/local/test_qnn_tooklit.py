# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.common.utils import retry_func, run_subprocess
from olive.logging import set_verbosity_debug

from ..utils import check_output, download_conda_installer, download_qc_toolkit, get_example_dir

set_verbosity_debug()


class TestQnnToolkit:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Download the qnn sdk and conda installer."""
        os.environ["QNN_SDK_ROOT"] = f"{download_qc_toolkit(tmp_path, 'qnn')}/opt/qcom/aistack"
        os.environ["CONDA_INSTALLER"] = download_conda_installer(tmp_path)

    def _setup_resource(self, use_olive_env):
        """Setups any state specific to the execution of the given module."""
        example_dir = get_example_dir("mobilenet")
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

        footprint = olive_run("raw_qnn_sdk_config.json", tempdir=os.environ.get("OLIVE_TEMPDIR", None))
        check_output(footprint)
