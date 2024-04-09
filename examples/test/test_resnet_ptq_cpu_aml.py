# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version
from utils import check_output, patch_config, set_azure_identity_logging

from olive.common.utils import retry_func, run_subprocess


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "resnet"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_model_data.py", "check": True})
    set_azure_identity_logging()

    yield
    os.chdir(cur_dir)


# TODO(myguo): consider split the test into two tests if the CredentialUnavailableError still happens in Windows.
@pytest.mark.parametrize(
    ("olive_json", "search_algorithm", "execution_order", "system"),
    [
        ("resnet_ptq_cpu.json", "random", "pass-by-pass", "aml_system"),
        ("resnet_ptq_cpu_aml_dataset.json", False, None, "local_system"),
        ("resnet_ptq_cpu_aml_dataset.json", False, None, "aml_system"),
    ],
)
@pytest.mark.skipif(
    version.parse(OrtVersion) == version.parse("1.16.0"),
    reason="resnet is not supported in ORT 1.16.0 caused by https://github.com/microsoft/onnxruntime/issues/17627",
)
def test_resnet(search_algorithm, execution_order, system, olive_json):
    # TODO(jambayk): add gpu e2e test
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system)

    footprint = olive_run(olive_config)
    check_output(footprint)
