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


def test_bert():
    from olive.workflows import run as olive_run

    generate_olive_workspace_config("olive-workspace-config.json")

    footprint = olive_run("bert_config.json")
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
