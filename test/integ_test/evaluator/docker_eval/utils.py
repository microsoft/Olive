# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path
from test.integ_test.utils import download_azure_blob
from zipfile import ZipFile

from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.systems.docker import DockerSystem, LocalDockerConfig


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir


current_dir, models_dir, data_dir = get_directories()
user_script = str(current_dir / "user_script.py")


def get_accuracy_metric(post_process, dataloader_func="create_dataloader"):
    accuracy_metric_config = {
        "user_script": user_script,
        "post_processing_func": post_process,
        "data_dir": data_dir,
        "dataloader_func": dataloader_func,
    }
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=[{"name": AccuracySubType.ACCURACY_SCORE}],
        user_config=accuracy_metric_config,
    )


def get_latency_metric(dataloader_func="create_dataloader"):
    latency_metric_config = {
        "user_script": user_script,
        "data_dir": data_dir,
        "dataloader_func": dataloader_func,
    }
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        user_config=latency_metric_config,
    )


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

    download_path = models_dir / "openvino.zip"
    openvino_model_config = {
        "container": "olivetest",
        "blob": "models/openvino.zip",
        "download_path": download_path,
    }
    download_azure_blob(**openvino_model_config)
    with ZipFile(download_path) as zip_ref:
        zip_ref.extractall(models_dir)
    return str(models_dir / "openvino")


def download_data():
    datasets.MNIST(data_dir, download=True, transform=ToTensor())


def get_huggingface_model():
    return {"hf_config": {"model_class": "AutoModelForSequenceClassification", "model_name": "prajjwal1/bert-tiny"}}


def get_pytorch_model():
    return {"model_path": str(models_dir / "model.pt")}


def get_onnx_model():
    return {"model_path": str(models_dir / "model.onnx")}


def get_openvino_model():
    return {"model_path": str(models_dir / "openvino")}


def delete_directories():
    shutil.rmtree(data_dir)
    shutil.rmtree(models_dir)


def get_docker_target():
    local_docker_config = LocalDockerConfig(
        image_name="olive",
        build_context_path=str(current_dir / "dockerfile"),
        dockerfile="Dockerfile",
    )
    return DockerSystem(local_docker_config=local_docker_config, is_dev=True)
