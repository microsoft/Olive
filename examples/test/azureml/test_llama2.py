# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import uuid

import pytest

from olive.common.constants import ACCOUNT_URL_TEMPLATE
from olive.common.utils import get_credentials

from ..utils import get_example_dir, get_gpu_compute, patch_config

account_name = os.environ.get("PIPELINE_TEST_ACCOUNT_NAME")
container_name = os.environ.get("PIPELINE_TEST_CONTAINER_NAME")


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("llama2"))


@pytest.mark.parametrize("sampler", [False])
@pytest.mark.parametrize("execution_order", [None])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize("cache_config", [None, {"account_name": account_name, "container_name": container_name}])
@pytest.mark.parametrize("olive_json", ["llama2_qlora.json"])
def test_llama2(sampler, execution_order, system, cache_config, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, sampler, execution_order, system, is_gpu=False, hf_token=True)

    # reduce qlora steps for faster test
    olive_config["passes"]["f"]["training_args"]["max_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["logging_steps"] = 5
    olive_config["passes"]["f"]["training_args"]["per_device_train_batch_size"] = 2
    olive_config["passes"]["f"]["training_args"]["per_device_eval_batch_size"] = 2

    # don't know what version of ort might be in the container, set to numpy for backward compatibility
    olive_config["passes"]["e"]["save_format"] = "numpy"

    # add shared cache config
    olive_config["cache_config"] = cache_config

    olive_config["systems"]["aml_system"] = get_gpu_compute(True)
    olive_config["systems"]["aml_system"]["datastores"] = container_name

    # set workflow host
    olive_config["workflow_host"] = "aml_system"

    # set a random workflow id, otherwise the test will fail due to aml job cache
    workflow_id = str(uuid.uuid4())
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

    account_url = ACCOUNT_URL_TEMPLATE.format(account_name=account_name)
    return ContainerClient(account_url=account_url, container_name=container_name, credential=get_credentials())
