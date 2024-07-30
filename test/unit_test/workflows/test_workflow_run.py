from test.unit_test.utils import (
    get_pytorch_model,
    get_pytorch_model_config,
    get_pytorch_model_io_config,
    pytorch_model_loader,
)
from unittest.mock import patch

import pytest

from olive.data.registry import Registry
from olive.hardware.accelerator import AcceleratorSpec
from olive.workflows import run as olive_run


@Registry.register_dataloader()
def workflow_run_test_dataloader(dataset, batch_size):
    return None


INPUT_MODEL_CONFIG = {
    "type": "PyTorchModel",
    "config": {
        "model_loader": pytorch_model_loader,
        "io_config": get_pytorch_model_io_config(),
    },
}

EVALUATORS_CONFIG = {
    "metrics": [
        {
            "name": "latency",
            "type": "latency",
            "sub_types": [
                {
                    "name": "avg",
                },
            ],
        }
    ]
}

DATA_CONFIGS = [
    {
        "name": "qat_train_data_config",
        "type": "DummyDataContainer",
        "load_dataset_config": {"type": "simple_dataset", "params": {"data_dir": "data"}},
        "pre_process_data_config": {"type": "skip_pre_process"},
        "post_process_data_config": {"type": "skip_post_process"},
        "dataloader_config": {"type": "workflow_run_test_dataloader", "params": {"batch_size": 100}},
    },
    {
        "name": "qat_val_data_config",
        "type": "DummyDataContainer",
        "load_dataset_config": {"type": "simple_dataset", "params": {"data_dir": "data"}},
        "pre_process_data_config": {"type": "skip_pre_process"},
        "post_process_data_config": {"type": "skip_post_process"},
        "dataloader_config": {"type": "workflow_run_test_dataloader", "params": {"batch_size": 100}},
    },
]

PASS_CONFIG = {
    "qat": {
        "type": "QuantizationAwareTraining",
        "config": {
            "train_data_config": "qat_train_data_config",
            "val_data_config": "qat_val_data_config",
            "num_epochs": 1,
            "modules_to_fuse": [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]],
            "qconfig_func": "create_qat_config",
        },
    },
}


@pytest.mark.parametrize(
    "config_test",
    [
        {
            "input_model": INPUT_MODEL_CONFIG,
            "data_configs": DATA_CONFIGS,
            "evaluators": {"common_evaluator": EVALUATORS_CONFIG},
            "passes": PASS_CONFIG,
            "engine": {"evaluator": "common_evaluator"},
        },
        {
            "input_model": INPUT_MODEL_CONFIG,
            "data_configs": DATA_CONFIGS,
            "passes": PASS_CONFIG,
            "engine": {"evaluator": EVALUATORS_CONFIG},
        },
    ],
)
@patch("olive.passes.pytorch.quantization_aware_training.QuantizationAwareTraining._run_for_config")
@patch("olive.systems.local.ModelConfig.from_json")
@patch("olive.engine.engine.ModelConfig.to_json")
def test_run_without_ep(mock_model_to_json, mock_model_from_json, mock_run, config_test, tmp_path):
    user_script = tmp_path / "user_script.py"
    with user_script.open("w"):
        pass

    config = config_test
    config["passes"]["qat"]["config"]["user_script"] = str(user_script)
    config["engine"]["cache_dir"] = str(tmp_path / "cache")
    config["engine"]["output_dir"] = str(tmp_path / "output")

    mock_run.return_value = get_pytorch_model()
    mock_model_from_json.return_value = get_pytorch_model_config()
    mock_model_to_json.return_value = {"type": "PyTorchModel", "config": {"io_config": {}}}
    ret = olive_run(config)
    assert len(ret) == 1
    assert next(iter(ret)) == AcceleratorSpec("cpu")
