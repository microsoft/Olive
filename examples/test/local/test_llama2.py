# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest

from olive.common.hf.login import huggingface_login

from ..utils import assert_nodes, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("llama2"))


@pytest.mark.parametrize("search_algorithm", [False])
@pytest.mark.parametrize("execution_order", [None])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["llama2_qlora.json"])
def test_llama2(search_algorithm, execution_order, system, olive_json):
    from olive.workflows import run as olive_run

    hf_token = os.environ.get("HF_TOKEN")
    huggingface_login(hf_token)

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system)

    # reduce qlora steps for faster test
    olive_config["passes"]["f"]["training_args"]["max_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["logging_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["per_device_train_batch_size"] = 2
    olive_config["passes"]["f"]["training_args"]["per_device_eval_batch_size"] = 2

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    assert_nodes(footprint)
