# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import zipfile
from pathlib import Path

import pytest
from utils import check_output, download_azure_blob

from olive.common.utils import retry_func, run_subprocess


def download_qnn_sdk():
    """Download the qnn sdk."""
    download_azure_blob(
        container="qualcomm-sdk",
        blob="qnn_snpe_sdk_linux.zip",
        download_path="qnn_snpe_sdk_linux.zip",
    )
    target_path = (Path(__file__).resolve() / "qnn_snpe_sdk_linux").resolve()
    target_path.mkdir(parents=True, exist_ok=True)
    run_subprocess(cmd="unzip qnn_snpe_sdk_linux.zip ", capture_output=True, check=True)
    with zipfile.ZipFile("qnn_snpe_sdk_linux.zip", "r") as zip_ref:
        zip_ref.extractall(target_path)
    return str(target_path)


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "mobilenet_qnn_qualcomm_npu"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python download_files.py", "check": True})
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_config.py --use_raw_qnn_sdk", "check": True})
    os.environ["QNN_SDK_ROOT"] = download_qnn_sdk()
    yield
    os.chdir(cur_dir)


def test_mobilenet_qnn():
    from olive.workflows import run as olive_run

    footprint = olive_run("raw_qnn_sdk_config")
    check_output(footprint)
