# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
from pathlib import Path

from utils import check_output, download_azure_blob

from olive.common.utils import retry_func, run_subprocess


def download_qnn_sdk(target_path=None):
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
    if not target_path:
        target_path = Path().resolve() / "qnn_sdk"
    target_path.mkdir(parents=True, exist_ok=True)
    if platform.system() == "Windows":
        cmd = f"powershell Expand-Archive -Path {download_path} -DestinationPath {str(target_path)}"
        run_subprocess(cmd=cmd, check=True)
    elif platform.system() == "Linux":
        run_subprocess(cmd=f"unzip {download_path} -d {str(target_path)}", check=True)
    return target_path


def setup_resource():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "mobilenet_qnn_qualcomm_npu"
    os.chdir(example_dir)

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
    retry_func(run_subprocess, kwargs={"cmd": "python download_files.py", "check": True})
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_config.py --use_raw_qnn_sdk", "check": True})


def test_mobilenet_qnn(tmp_path):
    from olive.workflows import run as olive_run

    os.environ["QNN_SDK_ROOT"] = str(download_qnn_sdk(tmp_path) / "opt" / "qcom" / "aistack")
    setup_resource()

    footprint = olive_run("raw_qnn_sdk_config.json")
    check_output(footprint)
