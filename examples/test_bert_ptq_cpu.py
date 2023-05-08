# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent
    example_dir = cur_dir / "bert_ptq_cpu"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


def check_output(footprint):
    footprint = footprint[0]
    assert footprint.nodes is not None
    for v in footprint.nodes.values():
        assert all([value > 0 for value in v.metrics.value.values()])


# Skip docker_system test until bug is fixed: https://github.com/docker/docker-py/issues/3113
@pytest.mark.parametrize("search_algorithm", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint"])
@pytest.mark.parametrize("system", ["local_system", "aml_system"])
@pytest.mark.parametrize("olive_json", ["bert_config.json"])
def test_bert(search_algorithm, execution_order, system, olive_json):
    # TODO: add gpu e2e test
    # if system == "docker_system" and platform.system() == "Windows":
    #     pytest.skip("Skip Linux containers on Windows host test case.")

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
    # set docker_system as dev
    # olive_config["systems"]["docker_system"]["config"]["is_dev"] = True

    # update host and target
    olive_config["engine"]["host"] = system if system != "docker_system" else "local_system"
    olive_config["engine"]["target"] = system

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
