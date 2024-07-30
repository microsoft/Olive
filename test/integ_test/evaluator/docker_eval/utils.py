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

from olive.common.config_utils import validate_config
from olive.data.config import DataComponentConfig, DataConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric, MetricType
from olive.systems.docker import DockerSystem, LocalDockerConfig

# pylint: disable=redefined-outer-name


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    user_script = current_dir / "user_script.py"

    return current_dir, models_dir, data_dir, user_script


_current_dir, _models_dir, _data_dir, _user_script = get_directories()


def _get_metric_data_config(name, dataset, post_process=None):
    data_config = DataConfig(
        name=name,
        user_script=str(_user_script),
        load_dataset_config=DataComponentConfig(
            type=dataset,
            params={"data_dir": str(_data_dir)},
        ),
    )
    if post_process:
        data_config.post_process_data_config = DataComponentConfig(type=post_process)
    return validate_config(data_config, DataConfig)


def get_accuracy_metric(post_process, dataset="mnist_dataset_for_docker_eval"):
    sub_types = [{"name": AccuracySubType.ACCURACY_SCORE, "metric_config": {"task": "multiclass", "num_classes": 10}}]
    return Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        data_config=_get_metric_data_config("accuracy_metric_data_config", dataset, post_process),
    )


def get_latency_metric(dataset="mnist_dataset_for_docker_eval"):
    return Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
        data_config=_get_metric_data_config("latency_metric_data_config", dataset),
    )


def download_models():
    pytorch_model_config = {
        "container": "olivetest",
        "blob": "models/model.pt",
        "download_path": str(_models_dir / "model.pt"),
    }
    download_azure_blob(**pytorch_model_config)

    onnx_model_config = {
        "container": "olivetest",
        "blob": "models/model.onnx",
        "download_path": str(_models_dir / "model.onnx"),
    }
    download_azure_blob(**onnx_model_config)

    download_path = str(_models_dir / "openvino.zip")
    openvino_model_config = {
        "container": "olivetest",
        "blob": "models/openvino.zip",
        "download_path": download_path,
    }
    download_azure_blob(**openvino_model_config)
    with ZipFile(download_path) as zip_ref:
        zip_ref.extractall(_models_dir)
    return str(_models_dir / "openvino")


def download_data():
    datasets.MNIST(str(_data_dir), download=True, transform=ToTensor())


def get_huggingface_model():
    return {
        "model_path": "hf-internal-testing/tiny-random-BertForSequenceClassification",
        "task": "text-classification",
    }


def get_pytorch_model():
    return {"model_path": str(_models_dir / "model.pt")}


def get_onnx_model():
    return {"model_path": str(_models_dir / "model.onnx")}


def get_openvino_model():
    return {"model_path": str(_models_dir / "openvino")}


def delete_directories():
    shutil.rmtree(_data_dir)
    shutil.rmtree(_models_dir)


def get_docker_target():
    local_docker_config = LocalDockerConfig(
        image_name="olive",
        build_context_path=str(_current_dir / "dockerfile"),
        dockerfile="Dockerfile",
    )
    return DockerSystem(local_docker_config=local_docker_config, is_dev=True)
