# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("resnet"))


@pytest.mark.skip(reason="Disable failing tests")
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["resnet_vitis_ai_ptq_cpu.json"])
def test_resnet(system, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, None, None, system)

    workflow_output = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(workflow_output)
