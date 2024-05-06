# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import sys

# pylint: disable=broad-exception-raised


def check_output(footprints):
    """Check if the search output is valid."""
    assert footprints, "footprints is empty. The search must have failed for all accelerator specs."
    for footprint in footprints.values():
        assert footprint.nodes
        for v in footprint.nodes.values():
            assert all(metric_result.value > 0 for metric_result in v.metrics.value.values())


def patch_config(config_json_path: str, search_algorithm: str, execution_order: str, system: str, is_gpu: bool = False):
    """Load the config json file and patch it with the given search algorithm, execution order and system."""
    with open(config_json_path) as fin:
        olive_config = json.load(fin)
    # set default logger severity
    olive_config["engine"]["log_severity_level"] = 0
    # set clean cache
    olive_config["engine"]["clean_cache"] = True

    # update search strategy
    if not search_algorithm:
        olive_config["engine"]["search_strategy"] = False
    else:
        olive_config["engine"]["search_strategy"] = {
            "search_algorithm": search_algorithm,
            "execution_order": execution_order,
        }
        if search_algorithm in ("random", "tpe"):
            olive_config["engine"]["search_strategy"]["search_algorithm_config"] = {"num_samples": 3, "seed": 0}

    update_azureml_config(olive_config)
    if system == "aml_system":
        # set aml_system
        set_aml_system(olive_config, is_gpu=is_gpu)
        olive_config["engine"]["host"] = system
        olive_config["engine"]["target"] = system
    elif system == "docker_system":
        # set docker_system
        set_docker_system(olive_config)
        olive_config["engine"]["host"] = system
        olive_config["engine"]["target"] = system
        # reduce agent size for docker system

        # as our docker image is big, we need to reduce the agent size to avoid timeout
        # for the docker system test, we skip to search for transformers optimization as
        # it is tested in other olive system tests
        olive_config["passes"]["transformers_optimization"]["disable_search"] = True
        olive_config["engine"]["search_strategy"]["search_algorithm_config"]["num_samples"] = 2

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

    client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
    if client_id is None:
        raise Exception("Please set the environment variable MANAGED_IDENTITY_CLIENT_ID")

    olive_config["azureml_client"] = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        # pipeline agents have multiple managed identities, so we need to specify the client_id
        "default_auth_params": {"managed_identity_client_id": client_id},
    }


def set_aml_system(olive_config, is_gpu=False):
    """Set the aml_system in the olive config."""
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    if is_gpu:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "config": {
                "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
                "aml_compute": "gpu-cluster",
                "aml_docker_config": {
                    "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.6-cudnn8-ubuntu20.04",
                    "conda_file_path": "conda_gpu.yaml",
                },
                "is_dev": True,
            },
        }

    else:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "config": {
                "accelerators": [{"device": "CPU", "execution_providers": ["CPUExecutionProvider"]}],
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
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    olive_config["systems"]["docker_system"] = {
        "type": "Docker",
        "config": {
            "accelerators": [{"device": "CPU", "execution_providers": ["CPUExecutionProvider"]}],
            "local_docker_config": {
                "image_name": "olive-image",
                "build_context_path": "docker",
                "dockerfile": "Dockerfile",
            },
            "is_dev": True,
        },
    }


def download_azure_blob(container, blob, download_path, storage_account="olivewheels"):
    from azure.identity import ManagedIdentityCredential
    from azure.storage.blob import BlobClient

    blob = BlobClient.from_blob_url(
        f"https://{storage_account}.blob.core.windows.net/{container}/{blob}",
        credential=ManagedIdentityCredential(client_id=os.environ.get("MANAGED_IDENTITY_CLIENT_ID")),
    )

    with open(download_path, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)


def set_azure_identity_logging():
    identity_logger = logging.getLogger("azure.identity")
    identity_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    identity_logger.addHandler(handler)
