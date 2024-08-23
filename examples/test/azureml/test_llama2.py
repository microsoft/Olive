# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import uuid

import pytest

from olive.common.utils import get_credentials

from ..utils import get_example_dir, get_gpu_compute, patch_config

account_url = os.environ.get("PIPELINE_TEST_ACCOUNT_URL")
container_name = os.environ.get("PIPELINE_TEST_CONTAINER_NAME")


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("llama2"))


@pytest.mark.parametrize("search_algorithm", [False])
@pytest.mark.parametrize("execution_order", [None])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("cloud_cache_config", [False, {"account_url": account_url, "container_name": container_name}])
@pytest.mark.parametrize("olive_json", ["llama2_qlora.json"])
def test_llama2(search_algorithm, execution_order, system, cloud_cache_config, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system, is_gpu=False, hf_token=True)

    # reduce qlora steps for faster test
    olive_config["passes"]["f"]["training_args"]["max_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["logging_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["per_device_train_batch_size"] = 2
    olive_config["passes"]["f"]["training_args"]["per_device_eval_batch_size"] = 2

    # add cloud cache system
    olive_config["cloud_cache_config"] = cloud_cache_config

    olive_config["systems"]["aml_system"] = get_gpu_compute(True)
    olive_config["systems"]["aml_system"]["datastores"] = container_name

    # set workflow host
    olive_config["workflow_host"] = "aml_system"

    # set a random workflow id, otherwise the test will fail due to aml job cache
    workflow_id = uuid.uuid4()
    olive_config["workflow_id"] = workflow_id

    olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))

    # assert outputs
    container_client = get_blob_client()
    assert any(container_client.list_blobs(workflow_id))

    # delete outputs
    for blob in container_client.list_blobs(workflow_id):
        container_client.delete_blob(blob.name)


def get_blob_client():
    from azure.storage.blob import ContainerClient

    return ContainerClient(account_url=account_url, container_name=container_name, credential=get_credentials())
