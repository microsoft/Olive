# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os


def check_search_output(footprints):
    """Check if the search output is valid."""
    for footprint in footprints.values():
        assert footprint.nodes is not None
        for v in footprint.nodes.values():
            assert all([metric_result.value > 0 for metric_result in v.metrics.value.values()])


def check_no_search_output(outputs):
    for output in outputs.values():
        output_metrics = output["metrics"]
        for item in output_metrics.values():
            assert item.value > 0


def patch_config(config_json_path: str, search_algorithm: str, execution_order: str, system: str):
    """Load the config json file and patch it with the given search algorithm, execution order and system."""
    with open(config_json_path, "r") as fin:
        olive_config = json.load(fin)
    # set default logger severity
    olive_config["engine"]["log_severity_level"] = 0
    # set clean cache
    olive_config["engine"]["clean_cache"] = True

    # update search strategy
    olive_config["engine"]["search_strategy"]["search_algorithm"] = search_algorithm
    if search_algorithm == "random" or search_algorithm == "tpe":
        olive_config["engine"]["search_strategy"]["search_algorithm_config"] = {"num_samples": 3, "seed": 0}
    olive_config["engine"]["search_strategy"]["execution_order"] = execution_order

    # set aml_system
    if system == "aml_system":
        set_aml_system(olive_config)
        update_azureml_config(olive_config)
    # set docker_system
    elif system == "docker_system":
        set_docker_system(olive_config)

    # update host and target
    olive_config["engine"]["host"] = system if system != "docker_system" else "local_system"
    olive_config["engine"]["target"] = system

    return olive_config


def update_azureml_config(olive_config):
    """Update the azureml config in the olive config."""
    subscription_id = os.environ.get("WORKSPACE_SUBSCRIPTION_ID")
    if subscription_id is None:
        raise Exception("Please set the environment variable WORKSPACE_SUBSCRIPTION_ID")

    resource_group = os.environ.get("WORKSPACE_RESOURCE_GROUP")
    if resource_group is None:
        raise Exception("Please set the environment variable WORKSPACE_RESOURCE_GROUP")

    workspace_name = os.environ.get("WORKSPACE_NAME")
    if workspace_name is None:
        raise Exception("Please set the environment variable WORKSPACE_NAME")

    olive_config["azureml_client"] = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
    }


def set_aml_system(olive_config):
    """Set the aml_system in the olive config."""
    olive_config["systems"]["aml_system"] = {
        "type": "AzureML",
        "config": {
            "accelerators": ["CPU"],
            "aml_compute": "cpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                "conda_file_path": "conda.yaml",
            },
            "is_dev": True,
        },
    }


def set_docker_system(olive_config):
    """Set the docker_system in the olive config."""
    olive_config["systems"]["docker_system"] = {
        "type": "Docker",
        "config": {
            "local_docker_config": {
                "image_name": "olive-image",
                "build_context_path": "docker",
                "dockerfile": "Dockerfile",
            },
            "is_dev": True,
        },
    }
