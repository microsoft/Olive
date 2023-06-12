# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.evaluator.metric import Metric, MetricType
from olive.evaluator.metric_config import MetricGoal
from olive.model import ONNXModel, PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxConversion, OnnxDynamicQuantization

ONNX_MODEL_PATH = Path(__file__).absolute().parent / "dummy_model.onnx"


class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.fc1 = nn.Linear(1, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        return x


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.randn(1), torch.rand(10)

    def __len__(self):
        return self.size


class FixedDummyDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.rng = np.random.default_rng(0)
        self.data = torch.tensor(self.rng.random((size, 1)))
        self.labels = torch.tensor(self.rng.random(1))

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]

    def __len__(self):
        return self.size


def pytorch_model_loader(model_path):
    return DummyModel().eval()


def get_pytorch_model():
    return PyTorchModel(
        model_loader=pytorch_model_loader,
        model_path=None,
        io_config={"input_names": ["input"], "output_names": ["output"], "input_shapes": [(1, 1)]},
    )


def get_pytorch_model_dummy_input():
    return torch.randn(1, 1)


def create_onnx_model_file():
    pytorch_model = pytorch_model_loader(model_path=None)
    dummy_input = get_pytorch_model_dummy_input()
    torch.onnx.export(
        pytorch_model, dummy_input, ONNX_MODEL_PATH, opset_version=10, input_names=["input"], output_names=["output"]
    )


def get_onnx_model():
    return ONNXModel(model_path=str(ONNX_MODEL_PATH))


def delete_onnx_model_files():
    if os.path.exists(ONNX_MODEL_PATH):
        os.remove(ONNX_MODEL_PATH)


def create_dataloader(datadir, batchsize):
    dataloader = DataLoader(DummyDataset(1))
    return dataloader


def create_fixed_dataloader(datadir, batchsize):
    dataloader = DataLoader(FixedDummyDataset(1))
    return dataloader


def get_accuracy_metric(*acc_subtype, random_dataloader=True, user_config=None, backend="torch_metrics"):
    accuracy_metric_config = {"dataloader_func": create_dataloader if random_dataloader else create_fixed_dataloader}
    accuracy_score_metric_config = {"mdmc_average": "global"}
    sub_types = [
        {
            "name": sub,
            "metric_config": accuracy_score_metric_config if sub == "accuracy_score" else {},
            "goal": MetricGoal(type="threshold", value=0.99),
        }
        for sub in acc_subtype
    ]
    sub_types[0]["priority"] = 1
    accuracy_metric = Metric(
        name="accuracy",
        type=MetricType.ACCURACY,
        sub_types=sub_types,
        user_config=user_config or accuracy_metric_config,
        backend=backend,
    )
    return accuracy_metric


def get_custom_metric():
    custom_metric = Metric(
        name="custom",
        type=MetricType.CUSTOM,
        sub_types=[{"name": "custom"}],
        user_config={"evaluate_func": "val", "user_script": "user_script"},
    )
    return custom_metric


def get_latency_metric(*lat_subtype, user_config=None):
    latency_metric_config = {"dataloader_func": create_dataloader}
    sub_types = [{"name": sub} for sub in lat_subtype]
    latency_metric = Metric(
        name="latency",
        type=MetricType.LATENCY,
        sub_types=sub_types,
        user_config=user_config or latency_metric_config,
    )
    return latency_metric


def get_onnxconversion_pass(ignore_pass_config=True):
    onnx_conversion_config = {}
    p = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
    if ignore_pass_config:
        return p
    pass_config = p.config_at_search_point({})
    pass_config = p.serialize_config(pass_config)
    return p, pass_config


def get_onnx_dynamic_quantization_pass(disable_search=False):
    p = create_pass_from_dict(OnnxDynamicQuantization, disable_search=disable_search)
    return p


def get_data_config():
    @Registry.register_dataset("test_dataset")
    def _test_dataset(test_value):
        ...

    @Registry.register_dataloader()
    def _test_dataloader(test_value):
        ...

    @Registry.register_pre_process()
    def _pre_process(test_value):
        ...

    @Registry.register_post_process()
    def _post_process(test_value):
        ...

    return DataConfig(
        components={
            "load_dataset": {
                "name": "test_dataset",
                "type": "test_dataset",
                "params": {"test_value": "test_value"},
            },
            "dataloader": {
                "name": "test_dataloader",
                "type": "_test_dataloader",  # This is the key to get dataloader
                "params": {"test_value": "test_value"},
            },
        }
    )


def get_glue_huggingface_data_config():
    return DataConfig(
        type="HuggingfaceContainer",
        params_config={
            "task": "text-classification",
            "model_name": "bert-base-uncased",
            "data_name": "glue",
            "subset": "mrpc",
            "split": "validation",
            "input_cols": ["sentence1", "sentence2"],
            "label_cols": ["label"],
            "batch_size": 1,
        },
    )


def get_dc_params_config():
    return DataConfig(
        params_config={
            "data_dir": "./params_config",
            "batch_size": 1,
            "label_cols": ["label_from_params_config"],
        },
        components={
            "load_dataset": DataComponentConfig(
                params={
                    "data_dir": "./params",
                    "batch_size": 10,
                }
            )
        },
    )
