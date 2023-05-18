# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import tempfile
from pathlib import Path

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.model import PyTorchModel
from olive.passes import OnnxConversion
from olive.passes.olive_pass import create_pass_from_dict
from olive.systems.azureml import AzureMLDockerConfig, AzureMLSystem


def test_aml_model():
    # ------------------------------------------------------------------
    # Azure ML System
    aml_compute = "cpu-cluster"
    folder_location = Path(__file__).absolute().parent
    conda_file_location = folder_location / "conda.yaml"
    workspace_config = get_olive_workspace_config()
    azureml_client_config = AzureMLClientConfig(**workspace_config)
    docker_config = AzureMLDockerConfig(
        base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file_path=conda_file_location,
    )
    aml_system = AzureMLSystem(
        azureml_client_config=azureml_client_config,
        aml_compute=aml_compute,
        aml_docker_config=docker_config,
        is_dev=True,
    )

    # ------------------------------------------------------------------
    # Input model
    pytorch_model = PyTorchModel(
        model_path={
            "type": "azureml_model",
            "config": {
                "azureml_client": azureml_client_config,
                "name": "bert_glue",
                "version": 10,
            },
        },
        io_config={
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "output_names": ["output"],
        },
    )

    # ------------------------------------------------------------------
    # Onnx conversion pass
    # config can be a dictionary
    onnx_conversion_config = {
        "target_opset": 13,
    }
    with tempfile.TemporaryDirectory() as tempdir:
        onnx_model_file = str(Path(tempdir) / "model.onnx")
        onnx_conversion_pass = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
        onnx_model = aml_system.run_pass(onnx_conversion_pass, pytorch_model, onnx_model_file)
        assert Path(onnx_model.model_path).is_file()


def get_olive_workspace_config():
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

    return workspace_config
