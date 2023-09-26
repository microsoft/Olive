# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_pytorch_model_dummy_input, pytorch_model_loader
from unittest.mock import MagicMock, patch

import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from olive.data.config import DataConfig
from olive.data.registry import Registry
from olive.resource_path import create_resource_path
from olive.workflows import run as olive_run


class DummyDataset(Dataset):
    def __init__(self, size):
        self.size = size

    def __getitem__(self, idx):
        return torch.randn(1), torch.rand(10).argmax()

    def __len__(self):
        return self.size


@Registry.register_dataset()
def dummy_dataset_dataroot(data_dir):
    return DummyDataset(1)


@Registry.register_post_process()
def post_processing_func(output):
    return output.argmax(axis=1)


def create_dataloader(datadir, batchsize, *args, **kwargs):
    return DataLoader(DummyDataset(1))


def get_dataloader_config():
    return {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "model_loader": pytorch_model_loader,
                "dummy_inputs_func": get_pytorch_model_dummy_input,
                "io_config": {"input_names": ["input"], "output_names": ["output"], "input_shapes": [(1, 1)]},
            },
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "accuracy",
                        "type": "accuracy",
                        "sub_types": [{"name": "accuracy_score", "priority": 1}],
                        "user_config": {
                            "data_dir": "data",
                            "dataloader_func": create_dataloader,
                            "batch_size": 16,
                            "post_processing_func": post_processing_func,
                        },
                    }
                ]
            }
        },
        "passes": {
            "onnx_conversion": {"type": "OnnxConversion", "config": {"target_opset": 13}},
            "perf_tuning": {
                "type": "OrtPerfTuning",
                "config": {
                    "dataloader_func": create_dataloader,
                    "batch_size": 16,
                    "data_dir": "data",
                },
            },
        },
        "engine": {
            "search_strategy": {"execution_order": "joint", "search_algorithm": "exhaustive"},
            "evaluator": "common_evaluator",
            "clean_cache": True,
            "output_dir": "./cache",
            "cache_dir": "./cache",
        },
    }


def get_data_config():
    return {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "model_loader": pytorch_model_loader,
                "dummy_inputs_func": get_pytorch_model_dummy_input,
                "io_config": {"input_names": ["input"], "output_names": ["output"], "input_shapes": [(1, 1)]},
            },
        },
        "data_configs": {
            "test_data_config": DataConfig(
                components={
                    "load_dataset": {
                        "name": "dummy_dataset_dataroot",
                        "type": "dummy_dataset_dataroot",
                        "params": {"data_dir": "data"},
                    },
                    "post_process_data": {
                        "type": "post_processing_func",
                    },
                }
            ),
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "accuracy",
                        "type": "accuracy",
                        "sub_types": [{"name": "accuracy_score", "priority": 1}],
                        # reference to data_config defined in global data_configs
                        "data_config": "test_data_config",
                    }
                ]
            }
        },
        "passes": {
            "onnx_conversion": {"type": "OnnxConversion", "config": {"target_opset": 13}},
            "perf_tuning": {
                "type": "OrtPerfTuning",
                "config": {
                    # "data_config": "test_data_config"
                    # This is just demo purpose to show how to use data_config in passes
                    "data_config": DataConfig(
                        components={
                            "load_dataset": {
                                "name": "dummy_dataset_dataroot",
                                "type": "dummy_dataset_dataroot",
                                "params": {"data_dir": "perfdata"},
                            },
                            "post_process_data": {
                                "type": "post_processing_func",
                            },
                        }
                    )
                },
            },
        },
        "engine": {
            "search_strategy": {"execution_order": "joint", "search_algorithm": "exhaustive"},
            "evaluator": "common_evaluator",
            "clean_cache": True,
            "output_dir": "./cache",
            "cache_dir": "./cache",
        },
    }


def concat_data_dir(data_root, data_dir):
    if data_root is None:
        data_dir = data_dir
    elif data_root.startswith("azureml://"):
        data_dir = data_root + "/" + data_dir
    else:
        data_dir = str(Path(data_root) / data_dir)

    return data_dir


@pytest.fixture(params=[None, "azureml://CIFAR-10/1", "local"])
def config(tmpdir, request):
    config_obj = get_dataloader_config()

    data_root = request.param
    if data_root is not None:
        if data_root == "local":
            tmpdir.mkdir("data")
            data_root = str(tmpdir)
        config_obj["data_root"] = data_root

    return config_obj


@patch("olive.cache.get_local_path")
@pytest.mark.parametrize("is_cmdline", [True, False])
def test_data_root_for_dataloader_func(mock_get_local_path, config, is_cmdline):
    mock_get_local_path.side_effect = lambda x, cache_dir: x.get_path()
    if is_cmdline:
        data_root = config.pop("data_root", None)
        best = olive_run(config, data_root=data_root)
    else:
        data_root = config.get("data_root")
        best = olive_run(config)

    data_dir = concat_data_dir(data_root, "data")
    mock_get_local_path.assert_called_with(create_resource_path(data_dir), ".olive-cache")
    assert best is not None


@pytest.fixture(params=[None, "azureml://CIFAR-10/1", "local"])
def data_config(tmpdir, request):
    config_obj = get_data_config()

    data_root = request.param
    if data_root is not None:
        if data_root == "local":
            tmpdir.mkdir("data")
            tmpdir.mkdir("perfdata")
            data_root = str(tmpdir)
        config_obj["data_root"] = data_root

    return config_obj


@patch("olive.cache.get_local_path")
def test_data_root_for_dataset(mock_get_local_path, data_config):
    mock_get_local_path.side_effect = lambda x, cache_dir: x.get_path()

    config = data_config
    data_root = config.get("data_root")

    mock = MagicMock(side_effect=dummy_dataset_dataroot)
    Registry.register_dataset("dummy_dataset_dataroot")(mock)
    best = olive_run(config)
    mock.assert_called_with(data_dir=concat_data_dir(data_root, "data"))

    data_dir_expected = concat_data_dir(data_root, "perfdata")
    mock.assert_any_call(data_dir=data_dir_expected)
    assert best is not None
