import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.workflows import run as olive_run
from test.utils import (
    get_pytorch_model,
    get_pytorch_model_config,
    get_pytorch_model_io_config,
    pytorch_model_loader,
)

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

PASS_CONFIG = {
    "gptq": {"type": "Gptq"},
}


@pytest.mark.parametrize(
    "config_test",
    [
        {
            "input_model": INPUT_MODEL_CONFIG,
            "evaluators": {"common_evaluator": EVALUATORS_CONFIG},
            "passes": PASS_CONFIG,
            "engine": {"evaluator": "common_evaluator"},
        },
        {
            "input_model": INPUT_MODEL_CONFIG,
            "passes": PASS_CONFIG,
            "engine": {"evaluator": EVALUATORS_CONFIG},
        },
    ],
)
@patch("olive.passes.pytorch.gptq.Gptq._run_for_config")
@patch("olive.systems.local.ModelConfig.from_json")
@patch("olive.engine.engine.ModelConfig.to_json")
def test_run_without_ep(mock_model_to_json, mock_model_from_json, mock_run, config_test, tmp_path):
    config = deepcopy(config_test)
    config["engine"]["cache_dir"] = str(tmp_path / "cache")
    config["engine"]["output_dir"] = str(tmp_path / "output")

    mock_run.return_value = get_pytorch_model()
    mock_model_from_json.return_value = get_pytorch_model_config()
    mock_model_to_json.return_value = {"type": "PyTorchModel", "config": {"io_config": {}}}
    workflow_output = olive_run(config)
    assert len(workflow_output.get_available_devices()) == 1
    assert workflow_output.get_available_devices()[0] == "cpu"


ONNX_INPUT_CONFIG = {"type": "ONNXModel", "model_path": "model.onnx"}


@pytest.mark.parametrize(
    ("config_test", "is_ep_required"),
    [
        (
            {
                "input_model": ONNX_INPUT_CONFIG,
                "evaluators": {"common_evaluator": EVALUATORS_CONFIG},
                "evaluator": "common_evaluator",
            },
            True,
        ),
        ({"input_model": ONNX_INPUT_CONFIG}, False),
        (
            {
                "input_model": INPUT_MODEL_CONFIG,
                "evaluators": {"common_evaluator": EVALUATORS_CONFIG},
                "evaluator": "common_evaluator",
            },
            False,
        ),
        ({"input_model": INPUT_MODEL_CONFIG}, False),
    ],
)
@patch("olive.engine.engine.Engine.run")
def test_create_accelerator_only_eval(mock_run, config_test, is_ep_required):
    with patch.object(sys.modules[olive_run.__module__], "create_accelerators") as mock_create_accelerators:
        olive_run(config_test)
        assert mock_create_accelerators.call_args.kwargs["is_ep_required"] == is_ep_required


def test_run_packages():
    # setup
    config = {
        "input_model": INPUT_MODEL_CONFIG,
        "evaluators": {"common_evaluator": EVALUATORS_CONFIG},
        "passes": PASS_CONFIG,
        "engine": {"evaluator": "common_evaluator"},
    }

    # execute
    olive_run(config, list_required_packages=True)
    requirements_file_path = Path("olive_requirements.txt")

    # assert
    assert (requirements_file_path).exists()
    with (requirements_file_path).open() as f:
        file = f.read()
        assert file == "onnxruntime"

    # cleanup
    requirements_file_path.unlink()
