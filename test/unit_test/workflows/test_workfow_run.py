from test.unit_test.utils import create_dataloader, get_pytorch_model, get_pytorch_model_config, pytorch_model_loader
from unittest.mock import patch

from olive.hardware.accelerator import AcceleratorSpec
from olive.workflows import run as olive_run


@patch("olive.passes.pytorch.quantization_aware_training.QuantizationAwareTraining._run_for_config")
@patch("olive.systems.local.ModelConfig.from_json")
@patch("olive.engine.engine.ModelConfig.to_json")
def test_run_without_ep(mock_model_to_json, mock_model_from_json, mock_run, tmp_path):
    user_script = tmp_path / "user_script.py"
    with user_script.open("w"):
        pass

    config = {
        "input_model": {
            "type": "PyTorchModel",
            "config": {
                "model_loader": pytorch_model_loader,
                "io_config": {"input_names": ["input"], "output_names": ["output"], "input_shapes": [(1, 1)]},
            },
        },
        "evaluators": {
            "common_evaluator": {
                "metrics": [
                    {
                        "name": "latency",
                        "type": "latency",
                        "sub_types": [
                            {
                                "name": "avg",
                            },
                        ],
                        "user_config": {"dataloader_func": create_dataloader},
                    }
                ]
            }
        },
        "passes": {
            "qat": {
                "type": "QuantizationAwareTraining",
                "config": {
                    "user_script": str(user_script),
                    "train_data_dir": "data",
                    "val_data_dir": "data",
                    "num_epochs": 1,
                    "train_dataloader_func": "create_train_dataloader",
                    "train_batch_size": 100,
                    "modules_to_fuse": [["conv1", "bn1"], ["conv2", "bn2"], ["conv3", "bn3"]],
                    "qconfig_func": "create_qat_config",
                },
                "evaluator": "common_evaluator",
            },
        },
        "engine": {
            "cache_dir": str(tmp_path / "cache"),
            "output_dir": str(tmp_path / "output"),
        },
    }

    mock_run.return_value = get_pytorch_model()
    mock_model_from_json.return_value = get_pytorch_model_config()
    mock_model_to_json.return_value = {"type": "PyTorchModel", "config": {}}
    ret = olive_run(config)
    assert len(ret) == 1
    assert next(iter(ret)) == AcceleratorSpec("cpu")
