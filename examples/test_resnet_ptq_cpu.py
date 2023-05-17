# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest

from olive.common.utils import retry_func, run_subprocess


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent
    example_dir = cur_dir / "resnet_ptq_cpu"
    os.chdir(example_dir)

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_model_data.py", "check": True})

    yield
    os.chdir(cur_dir)


def check_output(footprints):
    for footprint in footprints.values():
        assert footprint.nodes is not None
        for v in footprint.nodes.values():
            assert all([value > 0 for value in v.metrics.value.values()])


@pytest.mark.parametrize("search_algorithm", ["random"])
@pytest.mark.parametrize("execution_order", ["pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system", "aml_system"])
@pytest.mark.parametrize("olive_json", ["resnet_config.json"])
def test_resnet(search_algorithm, execution_order, system, olive_json):
    # TODO: add gpu e2e test
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
    olive_config["engine"]["target"] = system

    if system == "aml_system":
        update_azureml_config(olive_config)

    footprint = olive_run(olive_config)
    check_output(footprint)


def update_azureml_config(olive_config):
    subscription_id = os.environ.get("WORKSPACE_SUBSCRIPTION_ID")
    if subscription_id is None:
        raise Exception("Please set the environment variable WORKSPACE_SUBSCRIPTION_ID")

    resource_group = os.environ.get("WORKSPACE_RESOURCE_GROUP")
    if resource_group is None:
        raise Exception("Please set the environment variable WORKSPACE_RESOURCE_GROUP")

    workspace_name = os.environ.get("WORKSPACE_NAME")
    if workspace_name is None:
        raise Exception("Please set the environment variable WORKSPACE_NAME")

    olive_config["azureml_client"]["subscription_id"] = subscription_id
    olive_config["azureml_client"]["resource_group"] = resource_group
    olive_config["azureml_client"]["workspace_name"] = workspace_name
