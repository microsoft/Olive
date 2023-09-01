# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.integ_test.utils import download_azure_blob

from torchvision import datasets
from torchvision.transforms import ToTensor

from olive.evaluator.metric import LatencySubType, Metric, MetricType


def get_directories():
    current_dir = Path(__file__).resolve().parent

    models_dir = current_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    data_dir = current_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)

    return current_dir, models_dir, data_dir


current_dir, models_dir, data_dir = get_directories()
user_script = str(current_dir / "user_script.py")


def get_latency_metric():
    latency_metric_config = {
        "user_script": user_script,
        "data_dir": str(data_dir),
        "dataloader_func": "create_dataloader",
    }
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=[{"name": LatencySubType.AVG}],
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


def get_onnx_model():
    return str(models_dir / "model.onnx")
