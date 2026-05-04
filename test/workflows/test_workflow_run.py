import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.workflows import run as olive_run
from olive.workflows.run.run import _classify_run_config_source
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
    assert workflow_output.from_device() == "cpu"


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
    with patch.object(sys.modules[olive_run.__module__], "create_accelerator") as mock_create_accelerator:
        olive_run(config_test)
        assert mock_create_accelerator.call_args.kwargs["is_ep_required"] == is_ep_required


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


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_recipe_result_success(mock_run_engine, mock_log_recipe_result):
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "task": "text-generation",
            "load_kwargs": {"attn_implementation": "eager"},
        },
        "systems": {
            "local_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]}],
            }
        },
        "engine": {"target": "local_system"},
        "passes": {"dynamic_quant": {"type": "OnnxDynamicQuantization"}},
    }
    expected_output = object()
    mock_run_engine.return_value = expected_output

    output = olive_run(
        config,
        recipe_telemetry_metadata={
            "recipe_name": "Quantize",
            "recipe_command": "Quantize",
            "recipe_source": "generated_cli",
            "recipe_format": "generated",
        },
    )

    assert output is expected_output
    mock_log_recipe_result.assert_called_once()
    assert mock_log_recipe_result.call_args.args[0] == "Quantize"
    assert mock_log_recipe_result.call_args.kwargs["success"] is True

    metadata = mock_log_recipe_result.call_args.kwargs["metadata"]
    assert metadata["recipe_command"] == "Quantize"
    assert metadata["recipe_source"] == "generated_cli"
    assert metadata["recipe_format"] == "generated"
    assert metadata["workflow_id"] == "default_workflow"
    assert metadata["input_model_type"] == "hfmodel"
    assert metadata["input_model_source"] == "string_name"
    assert metadata["model_task"] == "text-generation"
    assert metadata["target_system_type"] == "LocalSystem"
    assert metadata["target_device"] == "gpu"
    assert metadata["execution_provider"] == "CUDAExecutionProvider"
    assert metadata["execution_providers"] == "CUDAExecutionProvider"
    assert metadata["pass_types"] == "onnxdynamicquantization"
    assert metadata["pass_count"] == 1
    assert metadata["data_config_count"] == 0
    assert metadata["search_enabled"] is False
    assert metadata["package_config_provided"] is False
    assert metadata["is_ci"] is False
    assert metadata["recipe_hash"]
    assert metadata["input_model_name_hash"]


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_recipe_result_failure(mock_run_engine, mock_log_recipe_result):
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "task": "text-generation",
            "load_kwargs": {"attn_implementation": "eager"},
        },
        "passes": {"dynamic_quant": {"type": "OnnxDynamicQuantization"}},
    }
    mock_run_engine.side_effect = ValueError("recipe failed")

    with pytest.raises(ValueError, match="recipe failed"):
        olive_run(
            config,
            recipe_telemetry_metadata={
                "recipe_name": "Quantize",
                "recipe_command": "Quantize",
                "recipe_source": "generated_cli",
                "recipe_format": "generated",
            },
        )

    mock_log_recipe_result.assert_called_once()
    assert mock_log_recipe_result.call_args.args[0] == "Quantize"
    assert mock_log_recipe_result.call_args.kwargs["success"] is False
    assert mock_log_recipe_result.call_args.kwargs["exception_type"] == "ValueError"


def test_classify_run_config_source_handles_non_pathlike_object():
    assert _classify_run_config_source(object()) == ("config_object", "object")
