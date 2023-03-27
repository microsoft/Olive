# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib as imp
import json
import os
import sys
from pathlib import Path
from unittest.mock import patch

import pytest


@pytest.fixture()
def example_dir():
    return str(Path(__file__).resolve().parent / "resnet_ptq_cpu")


@pytest.fixture(autouse=True)
def setup(example_dir):
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)
    sys.path.append(example_dir)
    import prepare_model_data

    # reload the module if there are multiple same name module.
    imp.reload(prepare_model_data)

    test_args = "prepare_model_data.py".split()

    with patch.object(sys, "argv", test_args):
        prepare_model_data.main()

    yield
    os.chdir(cur_dir)
    sys.path.remove(example_dir)


def check_output(metrics):
    assert metrics is not None
    assert all([value > 0 for value in metrics])


@pytest.mark.parametrize("search_algorithm", ["random"])
@pytest.mark.parametrize("execution_order", ["pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system", "aml_system"])
@pytest.mark.parametrize("olive_json", ["resnet_config.json"])
def test_resnet(search_algorithm, execution_order, system, olive_json):

    from olive.workflows import run as olive_run

    olive_config = None
    with open(olive_json, "r") as fin:
        olive_config = json.load(fin)

    # update search strategy
    olive_config["engine"]["search_strategy"]["search_algorithm"] = search_algorithm
    if search_algorithm == "random" or search_algorithm == "tpe":
        olive_config["engine"]["search_strategy"]["search_algorithm_config"] = {"num_samples": 5, "seed": 0}
    olive_config["engine"]["search_strategy"]["execution_order"] = execution_order

    # set aml_system as dev
    olive_config["systems"]["aml_system"]["config"]["is_dev"] = True

    # update host and target
    olive_config["engine"]["host"] = system
    olive_config["evaluators"]["common_evaluator"]["target"] = system

    best_execution = olive_run(olive_config)
    check_output(best_execution["metric"])
