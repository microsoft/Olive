# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest

from olive.common.utils import get_credentials

from ..utils import get_example_dir, patch_config

account_url = os.environ.get("PIPELINE_TEST_ACCOUNT_NAME") or "https://olivestorageaccount.blob.core.windows.net"
container_name = os.environ.get("PIPELINE_TEST_CONTAINER_NAME") or "pipelinetest"


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("llama2"))


@pytest.mark.parametrize("search_algorithm", [False])
@pytest.mark.parametrize("execution_order", [None])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("cloud_cache_config", [False, {"account_url": account_url, "container_name": container_name}])
@pytest.mark.parametrize("olive_json", ["llama2_qlora.json"])
def test_bert(search_algorithm, execution_order, system, cloud_cache_config, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system, is_gpu=False, hf_token=True)

    # reduce qlora steps for faster test
    olive_config["passes"]["f"]["training_args"]["max_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["logging_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["per_device_train_batch_size"] = 2
    olive_config["passes"]["f"]["training_args"]["per_device_eval_batch_size"] = 2

    # add cloud cache system
    olive_config["cloud_cache_config"] = cloud_cache_config

    olive_config["systems"]["aml_system"] = {
            "type": "AzureML",
            "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
            "aml_compute": "gpu-cluster",
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                "conda_file_path": "conda_gpu.yaml",
            },
            "is_dev": True,
            "hf_token": True,
        }
    # set workflow host
    olive_config["workflow_host"] = "aml_system"

    workflow_id = "llama2_pipeline_test"
    olive_config["workflow_id"] = workflow_id

    olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))

    # asset outputs
    container_client = get_blob_client(account_url, container_name)
    assert any(container_client.list_blobs(workflow_id))

    # delete outputs
    for blob in container_client.list_blobs(workflow_id):
        container_client.delete_blob(blob.name)

def get_blob_client(account_url, container_name):
    from azure.storage.blob import ContainerClient

    return ContainerClient(account_url=account_url, container_name=container_name, credential=get_credentials())
