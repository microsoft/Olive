# pylint: disable=protected-access
import json
import sys
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from olive.telemetry.recipe_telemetry import (
    _NO_OVERRIDE,
    _build_recipe_hash,
    _classify_input_model_source,
    _classify_run_config_source,
    _extract_config_overrides,
)
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
@patch("olive.telemetry.recipe_telemetry.is_ci_environment", return_value=False)
def test_run_logs_recipe_result_success(_, mock_run_engine, mock_log_recipe_result):
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
    assert metadata["target_execution_provider"] == "CUDAExecutionProvider"
    assert metadata["target_execution_providers"] == "CUDAExecutionProvider"
    assert metadata["host_system_type"] == "LocalSystem"
    assert "host_device" not in metadata
    assert "host_execution_provider" not in metadata
    assert "host_execution_providers" not in metadata
    assert metadata["pass_types"] == "onnxdynamicquantization"
    assert metadata["pass_count"] == 1
    assert metadata["data_config_count"] == 0
    assert metadata["search_enabled"] is False
    assert metadata["package_config_provided"] is False
    assert metadata["is_ci"] is False
    assert metadata["recipe_hash"]
    assert "input_model_name_hash" not in metadata
    assert "config_overrides" not in metadata


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_config_overrides_when_recipe_metadata_provides_overrides(mock_run_engine, mock_log_recipe_result):
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "task": "text-generation",
        }
    }
    mock_run_engine.return_value = object()

    olive_run(
        config,
        recipe_telemetry_metadata={
            "recipe_name": "WorkflowRun",
            "config_overrides": {
                "input_model": {
                    "type": "HfModel",
                    "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                },
                "engine": {"target": "local_system"},
                "data_path": Path("data"),
            },
        },
    )

    metadata = mock_log_recipe_result.call_args.kwargs["metadata"]
    config_overrides = json.loads(metadata["config_overrides"])
    assert config_overrides["input_model"]["model_path"] == "Qwen/Qwen2.5-0.5B-Instruct"
    assert config_overrides["engine"]["target"] == "<reference>"
    assert config_overrides["data_path"] == "<resource>"


def test_missing_baseline_keys_are_not_reported_as_overrides():
    value = {"passes": {"optimize": {"type": "OrtTransformersOptimization"}}}
    baseline = {
        "passes": {"optimize": {"type": "OrtTransformersOptimization"}},
        "extra_dependencies": {"cpu": ["onnxruntime"]},
    }

    assert _extract_config_overrides(value, baseline) is _NO_OVERRIDE


@patch("olive.workflows.run.run.log_error")
@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_recipe_result_failure(mock_run_engine, mock_log_recipe_result, mock_log_error):
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
    assert "exception_type" not in mock_log_recipe_result.call_args.kwargs
    mock_log_error.assert_called_once()
    assert mock_log_error.call_args.kwargs["exception_type"] == "ValueError"
    assert "recipe failed" in mock_log_error.call_args.kwargs["exception_message"]


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_skips_recipe_result_when_recipe_telemetry_is_not_emitted(mock_run_engine, mock_log_recipe_result):
    expected_output = object()
    mock_run_engine.return_value = expected_output

    output = olive_run(
        {
            "input_model": {
                "type": "HfModel",
                "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
                "task": "text-generation",
            }
        },
        emit_recipe_telemetry=False,
    )

    assert output is expected_output
    mock_log_recipe_result.assert_not_called()


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.systems.system_config.SystemConfig.create_system")
def test_run_logs_single_parent_recipe_result_for_docker_host(mock_create_system, mock_log_recipe_result):
    expected_output = object()
    docker_system = Mock()

    def run_workflow(container_run_config):
        container_run_config.engine.host = container_run_config.engine.target
        return expected_output

    docker_system.run_workflow.side_effect = run_workflow
    mock_create_system.return_value = docker_system
    config = {
        "input_model": {"type": "ONNXModel", "model_path": "model.onnx"},
        "systems": {
            "docker_system": {
                "type": "Docker",
                "config": {
                    "dockerfile": "Dockerfile",
                    "build_context_path": "build_context",
                    "image_name": "test-image:latest",
                    "work_dir": "/olive-ws",
                },
            },
            "local_system": {"type": "LocalSystem"},
        },
        "engine": {"host": "docker_system", "target": "local_system"},
    }

    output = olive_run(config)

    assert output is expected_output
    mock_log_recipe_result.assert_called_once()
    metadata = mock_log_recipe_result.call_args.kwargs["metadata"]
    assert metadata["host_system_type"] == "Docker"


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_recipe_host_metadata_without_explicit_target(mock_run_engine, mock_log_recipe_result):
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "task": "text-generation",
            "load_kwargs": {"attn_implementation": "eager"},
        },
        "systems": {
            "host_system": {
                "type": "LocalSystem",
                "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
            }
        },
        "engine": {"host": "host_system"},
    }
    mock_run_engine.return_value = object()

    olive_run(
        config,
        recipe_telemetry_metadata={
            "recipe_name": "Quantize",
            "recipe_command": "Quantize",
            "recipe_source": "generated_cli",
            "recipe_format": "generated",
        },
    )

    metadata = mock_log_recipe_result.call_args.kwargs["metadata"]
    assert "target_system_type" not in metadata
    assert "target_device" not in metadata
    assert "target_execution_provider" not in metadata
    assert "target_execution_providers" not in metadata
    assert metadata["host_system_type"] == "LocalSystem"
    assert metadata["host_device"] == "cpu"
    assert metadata["host_execution_provider"] == "CPUExecutionProvider"
    assert metadata["host_execution_providers"] == "CPUExecutionProvider"


@patch("olive.workflows.run.run.log_recipe_result")
@patch("olive.workflows.run.run.run_engine")
def test_run_logs_package_config_overrides_when_package_config_provided(mock_run_engine, mock_log_recipe_result):
    config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "Qwen/Qwen2.5-0.5B-Instruct",
            "task": "text-generation",
            "load_kwargs": {"attn_implementation": "eager"},
        }
    }
    mock_run_engine.return_value = object()

    olive_run(
        config,
        package_config={
            "passes": {
                "AddOliveMetadata": {
                    "module_path": "olive.passes.onnx.add_metadata.AddOliveMetadata",
                    "supported_providers": ["CPUExecutionProvider"],
                }
            },
            "extra_dependencies": {"custom_accelerator": ["custom-package"]},
        },
        recipe_telemetry_metadata={
            "recipe_name": "Quantize",
            "recipe_command": "Quantize",
            "recipe_source": "generated_cli",
            "recipe_format": "generated",
        },
    )

    metadata = mock_log_recipe_result.call_args.kwargs["metadata"]
    assert metadata["package_config_provided"] is True
    package_config_overrides = json.loads(metadata["package_config_overrides"])
    assert package_config_overrides["passes"][0]["supported_providers"] == ["CPUExecutionProvider"]
    assert "module_path" not in package_config_overrides["passes"][0]
    assert package_config_overrides["extra_dependencies"]["custom_accelerator"] == ["custom-package"]


def test_classify_run_config_source_handles_non_pathlike_object():
    assert _classify_run_config_source(object()) == ("config_object", "object")


def test_classify_input_model_source_does_not_depend_on_local_filesystem(tmp_path, monkeypatch):
    assert _classify_input_model_source("Qwen/Qwen2.5-0.5B-Instruct") == "string_name"

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bert-base-uncased").mkdir()

    assert _classify_input_model_source("bert-base-uncased") == "string_name"
    assert _classify_input_model_source("./model.onnx") == "local_file"
    assert _classify_input_model_source("model.onnx") == "local_file"


def test_recipe_hash_does_not_depend_on_local_model_path_presence(tmp_path, monkeypatch):
    config = {
        "input_model": {"type": "HfModel", "config": {"model_path": "bert-base-uncased"}},
        "engine": {"output_dir": "output"},
    }
    recipe_hash = _build_recipe_hash(config)

    monkeypatch.chdir(tmp_path)
    (tmp_path / "bert-base-uncased").mkdir()

    assert _build_recipe_hash(config) == recipe_hash


def test_recipe_hash_handles_path_values():
    config = {
        "input_model": {"type": "HfModel", "config": {"model_path": Path("model")}},
        "custom_value": Path("custom"),
    }

    assert _build_recipe_hash(config)
