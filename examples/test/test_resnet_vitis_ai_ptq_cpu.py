# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version
from utils import check_output, patch_config

from olive.common.utils import retry_func, run_subprocess


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "resnet"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_model_data.py", "check": True})

    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("search_algorithm", ["exhaustive"])
@pytest.mark.parametrize("execution_order", ["pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system", "aml_system"])
@pytest.mark.parametrize("olive_json", ["resnet_vitis_ai_ptq_cpu.json"])
@pytest.mark.skipif(
    version.parse(OrtVersion) >= version.parse("1.16.0"),
    reason="VitisAIQuantization is not supported in ORT 1.16.0 with TensorsData",
)
def test_resnet(search_algorithm, execution_order, system, olive_json):
    # TODO: add gpu e2e test
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system)

    footprint = olive_run(olive_config)
    check_output(footprint)
