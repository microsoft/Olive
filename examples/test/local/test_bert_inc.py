# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os

import pytest

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("bert"))


@pytest.mark.parametrize(
    "olive_json",
    [
        "bert_inc_dynamic_ptq_cpu.json",
        "bert_inc_ptq_cpu.json",
        "bert_inc_smoothquant_ptq_cpu.json",
        "bert_inc_static_ptq_cpu.json",
    ],
)
def test_bert(olive_json):
    from olive.workflows import run as olive_run

    with open(olive_json) as f:
        olive_config = json.load(f)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
