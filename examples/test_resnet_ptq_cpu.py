# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest

from olive.common.utils import run_subprocess


@pytest.fixture()
def example_dir():
    return str(Path(__file__).resolve().parent / "resnet_ptq_cpu")


@pytest.fixture(autouse=True)
def setup(example_dir):
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent
    os.chdir(example_dir)

    # import prepare_model_data
    run_subprocess("python prepare_model_data.py")

    yield
    os.chdir(cur_dir)


def check_output(footprint):
    assert footprint.footprints is not None
    for v in footprint.footprints.values():
        assert all([value > 0 for value in v.metrics.value.values()])


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
        olive_config["engine"]["search_strategy"]["search_algorithm_config"] = {"num_samples": 3, "seed": 0}
    olive_config["engine"]["search_strategy"]["execution_order"] = execution_order

    # set aml_system as dev
    olive_config["systems"]["aml_system"]["config"]["is_dev"] = True

    # update host and target
    olive_config["engine"]["host"] = system
    olive_config["evaluators"]["common_evaluator"]["target"] = system

    if system == "aml_system":
        generate_olive_workspace_config("olive-workspace-config.json")

    footprint = olive_run(olive_config)
    check_output(footprint)


def generate_olive_workspace_config(workspace_config_path):
    subscription_id = os.environ.get("WORKSPACE_SUBSCRIPTION_ID")
    if subscription_id is None:
        raise Exception("Please set the environment variable WORKSPACE_SUBSCRIPTION_ID")

    resource_group = os.environ.get("WORKSPACE_RESOURCE_GROUP")
    if resource_group is None:
        raise Exception("Please set the environment variable WORKSPACE_RESOURCE_GROUP")

    workspace_name = os.environ.get("WORKSPACE_NAME")
    if workspace_name is None:
        raise Exception("Please set the environment variable WORKSPACE_NAME")

    workspace_config = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
    }

    json.dump(workspace_config, open(workspace_config_path, "w"))
