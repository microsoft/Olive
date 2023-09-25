# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from utils import check_output, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("search_algorithm", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint"])
@pytest.mark.parametrize("system", ["aml_system"])
@pytest.mark.parametrize("olive_json", ["bert_auto_gpu.json"])
def test_bert(search_algorithm, execution_order, system, olive_json):

    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system, is_gpu=True)
    # do not add tensorrt execution provider as our CI machine does not have tensorrt env
    olive_config["engine"]["execution_providers"] = ["CUDAExecutionProvider"]

    footprint = olive_run(olive_config)
    check_output(footprint)
