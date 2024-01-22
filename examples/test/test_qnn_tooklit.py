# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from utils import check_output, download_azure_blob

from olive.common.utils import retry_func, run_subprocess


def download_qnn_sdk():
    """Download the qnn sdk."""
    download_azure_blob(
        container="olivetest",
        blob="qnn_sdk_linux.zip",
        download_path="qnn_sdk_linux.zip",
    )
    target_path = Path().resolve()
    run_subprocess(cmd=f"unzip qnn_sdk_linux.zip -d {str(target_path)}", check=True)
    return target_path


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "mobilenet_qnn_qualcomm_npu"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    os.environ["QNN_SDK_ROOT"] = str(download_qnn_sdk() / "opt" / "qcom" / "aistack")
    run_subprocess(cmd="python -m olive.platform_sdk.qualcomm.configure --py_version 3.8 --sdk qnn", check=True)
    # install dependencies
    install_cmd = [
        str(Path(os.environ["QNN_SDK_ROOT"]) / "olive-pyenv" / "bin" / "python"),
        str(Path(os.environ["QNN_SDK_ROOT"]) / "bin" / "check-python-dependency"),
    ]
    run_subprocess(cmd="\n".join(install_cmd), check=True)
    retry_func(run_subprocess, kwargs={"cmd": "python download_files.py", "check": True})
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_config.py --use_raw_qnn_sdk", "check": True})
    yield
    os.chdir(cur_dir)


def test_mobilenet_qnn():
    from olive.workflows import run as olive_run

    footprint = olive_run("raw_qnn_sdk_config.json")
    check_output(footprint)
