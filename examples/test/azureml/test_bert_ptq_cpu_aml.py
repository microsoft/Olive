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


@pytest.mark.parametrize(
    "olive_test_knob",
    [
        # aml system test
        ("bert_ptq_cpu.json", "tpe", "joint", "aml_system"),
        # aml model test on local system
        ("bert_ptq_cpu_aml.json", "tpe", "joint", "local_system"),
        # aml model test in aml system
        ("bert_ptq_cpu_aml.json", False, None, "aml_system"),
    ],
)
def test_bert(olive_test_knob):
    # olive_config: (config_json_path, sampler, execution_order, system)
    # bert_ptq_cpu.json: use huggingface model id
    # bert_ptq_cpu_aml.json: use aml model path
    from olive.workflows import run as olive_run

    olive_config = patch_config(*olive_test_knob)
    metrics = olive_config["evaluators"]["common_evaluator"]["metrics"]
    metrics[0]["sub_types"][0].pop("goal", None)
    metrics[1]["sub_types"][0].pop("goal", None)

    output = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(output)
