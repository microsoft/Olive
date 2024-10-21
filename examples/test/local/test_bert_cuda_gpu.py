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


@pytest.mark.parametrize("search_algorithm", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint", "pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["bert_cuda_gpu.json"])
def test_bert(search_algorithm, execution_order, system, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system, is_gpu=True)
    olive_config["passes"]["session_params_tuning"]["enable_cuda_graph"] = False

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
