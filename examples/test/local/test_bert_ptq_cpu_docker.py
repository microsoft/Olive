# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform

import pytest

from olive.common.constants import OS

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("bert"))


@pytest.mark.parametrize("sampler", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint"])
@pytest.mark.parametrize("system", ["docker_system"])
@pytest.mark.parametrize("olive_json", ["bert_ptq_cpu.json"])
def test_bert(sampler, execution_order, system, olive_json):
    if system == "docker_system" and platform.system() == OS.WINDOWS:
        pytest.skip("Skip Linux containers on Windows host test case.")

    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, sampler, execution_order, system)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
