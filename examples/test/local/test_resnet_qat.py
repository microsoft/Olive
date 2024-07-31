# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest

from olive.common.utils import retry_func, run_subprocess

from ..utils import check_output, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent.parent
    example_dir = cur_dir / "resnet"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_model_data.py", "check": True})

    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("search_algorithm", ["random"])
@pytest.mark.parametrize("execution_order", ["pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize(
    "olive_json", ["resnet_qat_default_train_loop_cpu.json", "resnet_qat_lightning_module_cpu.json"]
)
def test_resnet(search_algorithm, execution_order, system, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
