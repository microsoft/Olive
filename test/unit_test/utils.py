# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from olive.evaluator.metric import Metric, MetricType
from olive.evaluator.metric_config import MetricGoal
from olive.model import ONNXModel, PyTorchModel
from olive.passes.onnx.conversion import OnnxConversion

ONNX_MODEL_PATH = Path(__file__).absolute().parent / "dummy_model.onnx"


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(10, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.randn(10), 1

    def __len__(self):
        return self.size


def pytorch_model_loader(model_path):
    return DummyModel().eval()


def get_pytorch_model():
    return PyTorchModel(model_loader=pytorch_model_loader, model_path=None)


def create_onnx_model_file():
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = torch.randn(1, 10)
    torch.onnx.export(
        pytorch_model, dummy_input, ONNX_MODEL_PATH, opset_version=10, input_names=["input"], output_names=["output"]
    )


def get_onnx_model():
    return ONNXModel(model_path=str(ONNX_MODEL_PATH))


def delete_onnx_model_files():
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)


def create_dataloader(datadir, batchsize):
    dataloader = DataLoader(DummyDataset(10))
    return dataloader


def get_accuracy_metric(acc_subtype):
    accuracy_metric_config = {"dataloader_func": create_dataloader}
    accuracy_metric = Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_type=acc_subtype,
        goal=MetricGoal(type="threshold", value=0.99),
        user_config=accuracy_metric_config,
    )
    return accuracy_metric


def get_custom_metric():
    custom_metric = Metric(
        name="custom",
        type=MetricType.CUSTOM,
        user_config={"evaluate_func": "val", "user_script": "user_script"},
    )
    return custom_metric


def get_latency_metric(lat_subtype):
    latency_metric_config = {"dataloader_func": create_dataloader}
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_type=lat_subtype,
        user_config=latency_metric_config,
    )
    return latency_metric


def get_onnxconversion_pass():
    onnx_conversion_config = {
        "input_names": ["input"],
        "output_names": ["output"],
    }
    p = OnnxConversion(onnx_conversion_config)
    return p
