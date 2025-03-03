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
    os.chdir(get_example_dir("bert"))


@pytest.mark.parametrize("sampler", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["bert_ptq_cpu.json"])
def test_bert(sampler, execution_order, system, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, sampler, execution_order, system)
    # remove the latency goal since it is flaky on CI
    metrics = olive_config["evaluators"]["common_evaluator"]["metrics"]
    del metrics[1]["sub_types"][0]["goal"]

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
