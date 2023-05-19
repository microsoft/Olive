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
    example_dir = cur_dir / "bert_hf_cpu_aml"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


def check_output(footprints):
    for footprint in footprints.values():
        assert footprint.nodes is not None
        for v in footprint.nodes.values():
            assert all([value > 0 for value in v.metrics.value.values()])


@pytest.mark.parametrize("system", ["local_system", "aml_system"])
def test_bert(system):
    from olive.workflows import run as olive_run

    olive_config = None
    with open("bert_config.json", "r") as fin:
        olive_config = json.load(fin)

    # update azureml config
    if system == "aml_system":
        update_azureml_config(olive_config)

    # update host and target
    olive_config["engine"]["host"] = system
    olive_config["engine"]["target"] = system

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
