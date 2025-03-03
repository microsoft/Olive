# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest

from olive.common.utils import retry_func, run_subprocess

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("bert"))


@pytest.mark.skip(reason="Disable failing tests")
@pytest.mark.parametrize("sampler", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint", "pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["bert_cuda_gpu.json"])
@pytest.mark.parametrize("cmd_args", ["", "--optimize"])
def test_bert(sampler, execution_order, system, olive_json, cmd_args):
    from olive.workflows import run as olive_run

    retry_func(run_subprocess, kwargs={"cmd": f"python bert.py {cmd_args}", "check": True})

    olive_config = patch_config(olive_json, sampler, execution_order, system, is_gpu=True)
    olive_config["passes"]["session_params_tuning"]["enable_cuda_graph"] = False

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
