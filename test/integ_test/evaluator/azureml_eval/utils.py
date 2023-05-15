# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import shutil
from pathlib import Path

from azure.storage.blob import BlobClient
from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.systems.azureml import AzureMLDockerConfig, AzureMLSystem


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir


current_dir, models_dir, data_dir = get_directories()
user_script = str(current_dir / "user_script.py")


def get_accuracy_metric():
    accuracy_metric_config = {
        "user_script": user_script,
        "post_processing_func": "post_process",
        "data_dir": data_dir,
        "dataloader_func": "create_dataloader",
    }
    accuracy_metric = Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_type=AccuracySubType.ACCURACY_SCORE,
        user_config=accuracy_metric_config,
    )
    return accuracy_metric


def get_latency_metric():
    latency_metric_config = {
        "user_script": user_script,
        "data_dir": data_dir,
        "dataloader_func": "create_dataloader",
    }
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=LatencySubType.AVG,
        user_config=latency_metric_config,
    )
    return latency_metric


def download_models():
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": models_dir / "model.pt",
    }
    download_azure_blob(**pytorch_model_config)

    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": models_dir / "model.onnx",
    }
    download_azure_blob(**onnx_model_config)


def download_data():
    datasets.MNIST(data_dir, download=True, transform=ToTensor())


def get_pytorch_model():
    return str(models_dir / "model.pt")


def get_onnx_model():
    return str(models_dir / "model.onnx")


def download_azure_blob(container, blob, download_path):
    try:
        conn_str = os.environ["OLIVEWHEELS_STORAGE_CONNECTION_STRING"]
    except KeyError:
        raise Exception("Please set the environment variable OLIVEWHEELS_STORAGE_CONNECTION_STRING")

    blob = BlobClient.from_connection_string(conn_str=conn_str, container_name=container, blob_name=blob)

    with open(download_path, "wb") as my_blob:
        blob_data = blob.download_blob()
        blob_data.readinto(my_blob)


def delete_directories():
    shutil.rmtree(data_dir)
    shutil.rmtree(models_dir)


def get_aml_target():
    aml_compute = "cpu-cluster"
    current_path = Path(__file__).absolute().parent
    conda_file_location = current_path / "conda.yaml"
    config_file_location = current_path / "olive-workspace-config.json"
    generate_olive_workspace_config(config_file_location)
    azureml_client_config = AzureMLClientConfig(aml_config_path=config_file_location)
    docker_config = AzureMLDockerConfig(
        base_image="mcr.microsoft.com/azureml/openmpi4.1.0-ubuntu20.04",
        conda_file_path=conda_file_location,
    )
    return AzureMLSystem(
        azureml_client_config=azureml_client_config,
        aml_compute=aml_compute,
        aml_docker_config=docker_config,
        is_dev=True,
    )


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
