# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from utils import check_search_output, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("search_algorithm", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("olive_json", ["bert_ptq_cpu.json"])
def test_bert(search_algorithm, execution_order, system, olive_json):

    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system)

    footprint = olive_run(olive_config)
    check_search_output(footprint)
