# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import platform
import sys

from olive.common.constants import OS
from olive.common.utils import run_subprocess

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
    olive_config["log_severity_level"] = 0
    # set clean cache
    olive_config["clean_cache"] = True

    # update search strategy
    if not search_algorithm:
        olive_config["search_strategy"] = False
    else:
        olive_config["search_strategy"] = {
            "search_algorithm": search_algorithm,
            "execution_order": execution_order,
        }
        if search_algorithm in ("random", "tpe"):
            olive_config["search_strategy"].update({"num_samples": 3, "seed": 0})

    update_azureml_config(olive_config)
    if system == "aml_system":
        # set aml_system
        set_aml_system(olive_config, is_gpu=is_gpu)
        olive_config["host"] = system
        olive_config["target"] = system
    elif system == "docker_system":
        # set docker_system
        set_docker_system(olive_config)
        olive_config["host"] = system
        olive_config["target"] = system
        # reduce agent size for docker system

        # as our docker image is big, we need to reduce the agent size to avoid timeout
        # for the docker system test, we skip to search for transformers optimization as
        # it is tested in other olive system tests
        olive_config["passes"]["transformers_optimization"]["disable_search"] = True
        olive_config["search_strategy"]["num_samples"] = 2

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

    exclude_managed_identity_credential = (
        {"exclude_managed_identity_credential": True} if "EXCLUDE_MANAGED_IDENTITY_CREDENTIAL" in os.environ else {}
    )

    client_id = os.environ.get("MANAGED_IDENTITY_CLIENT_ID")
    if client_id is None and not exclude_managed_identity_credential:
        raise Exception("Please set the environment variable MANAGED_IDENTITY_CLIENT_ID")

    olive_config["azureml_client"] = {
        "subscription_id": subscription_id,
        "resource_group": resource_group,
        "workspace_name": workspace_name,
        # pipeline agents have multiple managed identities, so we need to specify the client_id
        "default_auth_params": {"managed_identity_client_id": client_id, **exclude_managed_identity_credential},
    }


def set_aml_system(olive_config, is_gpu=False):
    """Set the aml_system in the olive config."""
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    if is_gpu:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
            "aml_compute": "gpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                "conda_file_path": "conda_gpu.yaml",
            },
            "is_dev": True,
        }

    else:
        olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "accelerators": [{"device": "CPU", "execution_providers": ["CPUExecutionProvider"]}],
            "aml_compute": "cpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
                "conda_file_path": "conda.yaml",
            },
            "is_dev": True,
        }


def set_docker_system(olive_config):
    """Set the docker_system in the olive config."""
    if "systems" not in olive_config:
        olive_config["systems"] = {}

    olive_config["systems"]["docker_system"] = {
        "type": "Docker",
        "accelerators": [{"device": "CPU", "execution_providers": ["CPUExecutionProvider"]}],
        "local_docker_config": {
            "image_name": "olive-image",
            "build_context_path": "docker",
            "dockerfile": "Dockerfile",
        },
        "is_dev": True,
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


def download_conda_installer(parent_dir):
    if platform.system() == OS.WINDOWS:
        conda_installer_blob, conda_installer_path = (
            "conda-installers/Miniconda3-latest-Windows-x86_64.exe",
            parent_dir / "conda_installer.exe",
        )
    elif platform.system() == OS.LINUX:
        conda_installer_blob, conda_installer_path = (
            "conda-installers/Miniconda3-latest-Linux-x86_64.sh",
            parent_dir / "conda_installer.sh",
        )
    else:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")

    download_azure_blob(
        container="olivetest",
        blob=conda_installer_blob,
        download_path=conda_installer_path,
    )

    return str(conda_installer_path)


def download_qc_toolkit(parent_dir, toolkit):
    blob, download_path = (
        f"{toolkit}_sdk_{platform.system().lower()}.zip",
        f"{toolkit}_sdk_{platform.system().lower()}.zip",
    )

    download_azure_blob(
        container="olivetest",
        blob=blob,
        download_path=download_path,
    )
    target_path = parent_dir / f"{toolkit}_sdk"
    target_path.mkdir(parents=True, exist_ok=True)
    if platform.system() == OS.WINDOWS:
        cmd = f"powershell Expand-Archive -Path {download_path} -DestinationPath {str(target_path)}"
        run_subprocess(cmd=cmd, check=True)
    elif platform.system() == OS.LINUX:
        run_subprocess(cmd=f"unzip {download_path} -d {str(target_path)}", check=True)

    return str(target_path)


def set_azure_identity_logging():
    identity_logger = logging.getLogger("azure.identity")
    identity_logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler(stream=sys.stdout)
    identity_logger.addHandler(handler)
