# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest
from utils import check_no_search_output, update_azureml_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


def test_bert():
    from olive.workflows import run as olive_run

    olive_config = None
    with open("bert_ptq_cpu_aml.json", "r") as fin:
        olive_config = json.load(fin)

    # set log severity to debug
    olive_config["engine"]["log_severity_level"] = 0
    # set cache clean to True
    olive_config["engine"]["clean_cache"] = True

    # update azureml config
    update_azureml_config(olive_config)

    output = olive_run(olive_config)
    check_no_search_output(output)
