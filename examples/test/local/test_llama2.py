# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest
from packaging import version

from ..utils import assert_nodes, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("llama2"))


@pytest.mark.parametrize("sampler", [False])
@pytest.mark.parametrize("execution_order", [None])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["llama2_qlora.json"])
def test_llama2(sampler, execution_order, system, olive_json):
    from onnxruntime import __version__ as ort_version

    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, sampler, execution_order, system)

    # replace meta-llama with open-llama version of the model
    # doesn't require login
    olive_config["input_model"]["model_path"] = "openlm-research/open_llama_7b_v2"

    # reduce qlora steps for faster test
    olive_config["passes"]["f"]["training_args"]["max_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["logging_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["per_device_train_batch_size"] = 2
    olive_config["passes"]["f"]["training_args"]["per_device_eval_batch_size"] = 2

    if version.parse(ort_version) < version.parse("1.20"):
        olive_config["passes"]["e"]["save_format"] = "numpy"

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    assert_nodes(footprint)
