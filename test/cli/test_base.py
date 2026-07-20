# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from olive.cli.base import get_input_model_config


@pytest.mark.parametrize(
    (
        "model_name_or_path",
        "trust_remote_code",
        "task",
        "model_script",
        "script_dir",
        "has_function_results",
        "expected_config",
    ),
    [
        # model_loader test
        (
            None,  # model_name_or_path
            False,  # trust_remote_code
            None,  # task
            "model.py",  # model_script
            "scripts",  # script_dir
            {"_model_loader": True, "_io_config": False, "_dummy_inputs": True},  # func_exist results
            {  # expected config
                "type": "PyTorchModel",
                "model_script": "model.py",
                "script_dir": "scripts",
                "model_loader": "_model_loader",
                "dummy_inputs_func": "_dummy_inputs",
            },
        ),
        # AML registry model test
        (
            "azureml://registries/my_registry/models/my_model/versions/1",  # model_name_or_path
            False,  # trust_remote_code
            "task",  # task
            None,  # model_script
            None,  # script_dir
            {"_model_loader": False, "_io_config": False, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "HfModel",
                "task": "task",
                "model_path": {
                    "type": "azureml_registry_model",
                    "registry_name": "my_registry",
                    "name": "my_model",
                    "version": "1",
                },
                "load_kwargs": {
                    "trust_remote_code": False,
                    "attn_implementation": "sdpa",
                },
            },
        ),
        # HF url test
        (
            "https://huggingface.co/my_model/my_model",  # model_name_or_path
            True,  # trust_remote_code
            "task",  # task
            None,  # model_script
            None,  # script_dir
            {"_model_loader": False, "_io_config": False, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "HfModel",
                "task": "task",
                "model_path": "my_model/my_model",
                "load_kwargs": {
                    "trust_remote_code": True,
                    "attn_implementation": "sdpa",
                },
            },
        ),
        # HF str name test
        (
            "hf_model",  # model_name_or_path
            False,  # trust_remote_code
            "task",  # task
            None,  # model_script
            None,  # script_dir
            {"_model_loader": False, "_io_config": False, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "HfModel",
                "task": "task",
                "model_path": "hf_model",
                "load_kwargs": {
                    "trust_remote_code": False,
                    "attn_implementation": "sdpa",
                },
            },
        ),
        # Local pt model test
        (
            "model.pt",  # model_name_or_path
            False,  # trust_remote_code
            None,  # task
            "model.py",  # model_script
            None,  # script_dir
            {"_model_loader": True, "_io_config": True, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "PyTorchModel",
                "model_script": "model.py",
                "model_loader": "_model_loader",
                "io_config": "_io_config",
                "model_path": "model.pt",
            },
        ),
        # Local onnx model test
        (
            "model.onnx",  # model_name_or_path
            False,  # trust_remote_code
            None,  # task
            None,  # model_script
            None,  # script_dir
            {"_model_loader": False, "_io_config": False, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "OnnxModel",
                "model_path": "model.onnx",
            },
        ),
        # Local hf model test
        (
            "hf",  # model_name_or_path
            False,  # trust_remote_code
            "task",  # task
            None,  # model_script
            None,  # script_dir
            {"_model_loader": False, "_io_config": False, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "type": "HfModel",
                "task": "task",
                "model_path": "hf",
                "load_kwargs": {
                    "trust_remote_code": False,
                    "attn_implementation": "sdpa",
                },
            },
        ),
    ],
)
@patch("olive.cli.base.UserModuleLoader")
def test_get_input_model_config(
    MockUserModuleLoader,
    model_name_or_path,
    trust_remote_code,
    task,
    model_script,
    script_dir,
    has_function_results,
    expected_config,
):
    # setup
    config = {}
    args = SimpleNamespace(
        model_name_or_path=model_name_or_path,
        trust_remote_code=trust_remote_code,
        task=task,
        model_script=model_script,
        script_dir=script_dir,
    )

    mock_instance = MockUserModuleLoader.return_value

    def mock_path_exists(path):
        return path == Path(model_name_or_path)

    def mock_path_is_dir(path):
        return path == Path(model_name_or_path) and "output_model" in path.name

    def mock_path_is_file(path):
        return path == Path(model_name_or_path) and path.suffix in (".pt", ".onnx")

    with (
        patch.object(Path, "exists", new=mock_path_exists),
        patch.object(Path, "is_dir", new=mock_path_is_dir),
        patch.object(Path, "is_file", new=mock_path_is_file),
    ):

        def has_function_side_effect(arg):
            return has_function_results.get(arg, False)

        mock_instance.has_function.side_effect = has_function_side_effect

        # execute
        config = get_input_model_config(args)

        # assert
        assert config == expected_config


@patch("olive.cli.base.UserModuleLoader")
def test_insert_input_model_pt_model_missing_loader(MockUserModuleLoader):
    # setup
    args = SimpleNamespace(
        model_name_or_path=None,
        trust_remote_code=False,
        task=None,
        model_script="model.py",
        script_dir="scripts",
    )
    MockUserModuleLoader.return_value.has_function.return_value = False

    # execute and assert
    with pytest.raises(ValueError, match=r"_model_loader function is required in model_script for PyTorch model\."):
        get_input_model_config(args)


def test_insert_input_model_invalid_hf_model_name():
    # setup
    args = SimpleNamespace(
        model_name_or_path="invalid-name",
        trust_remote_code=False,
        task=None,
        model_script=None,
        script_dir=None,
    )

    # execute and assert
    with pytest.raises(ValueError, match=r"invalid-name is not a valid Huggingface model name\."):
        get_input_model_config(args)


@patch("huggingface_hub.repo_exists", return_value=True)
def test_get_input_model_config_hf_test_model(_):
    args = SimpleNamespace(
        model_name_or_path="hf_model",
        trust_remote_code=False,
        task="text-generation",
        model_script=None,
        script_dir=None,
        test=True,
        output_path="out_dir",
    )

    config = get_input_model_config(args)

    assert config["test_model_config"] == {"hidden_layers": 2}
    assert config["test_model_path"] == str(Path("out_dir") / "reference_hf_model")


@patch("huggingface_hub.repo_exists", return_value=True)
def test_get_input_model_config_hf_test_model_requires_path_without_output_path(_):
    args = SimpleNamespace(
        model_name_or_path="hf_model",
        trust_remote_code=False,
        task="text-generation",
        model_script=None,
        script_dir=None,
        test=True,
    )

    with pytest.raises(ValueError, match=r"--test requires --output_path to store the generated reference model\."):
        get_input_model_config(args)


def test_insert_input_model_cli_output_model():
    # setup
    model_path = str(Path(__file__).parent.resolve() / "output_model")
    args = SimpleNamespace(
        model_name_or_path=model_path,
        trust_remote_code=False,
        task=None,
        model_script=None,
        script_dir=None,
    )
    expected_config = {"type": "PyTorchModel", "model_path": "model_path"}

    # execute
    config = get_input_model_config(args)

    # assert
    assert config == expected_config


def test_get_input_model_config_rewrites_stale_model_path(tmp_path):
    """Test that model_path is rewritten when the stored path is from a different machine/location."""
    onnx_file_name = "model.onnx"
    # Create the ONNX file locally in tmp_path (content doesn't matter, only existence is checked)
    (tmp_path / onnx_file_name).touch()

    # Write model_config.json with a stale absolute model_path from a different location
    stale_model_path = "/some/other/machine/path/to/model"
    model_config = {
        "type": "OnnxModel",
        "config": {
            "model_path": stale_model_path,
            "onnx_file_name": onnx_file_name,
        },
    }
    (tmp_path / "model_config.json").write_text(json.dumps(model_config))

    args = SimpleNamespace(
        model_name_or_path=str(tmp_path),
        trust_remote_code=False,
        task=None,
        model_script=None,
        script_dir=None,
    )

    # execute
    config = get_input_model_config(args)

    # assert that model_path has been rewritten to the current (local) directory
    assert config["config"]["model_path"] == str(tmp_path)
    assert config["config"]["onnx_file_name"] == onnx_file_name


def test_get_input_model_config_no_crash_without_onnx_file_name(tmp_path):
    """Test that get_input_model_config does not crash when onnx_file_name is missing from config."""
    stale_model_path = "/some/other/machine/path/to/model"
    model_config = {
        "type": "OnnxModel",
        "config": {
            "model_path": stale_model_path,
            # onnx_file_name intentionally omitted
        },
    }
    (tmp_path / "model_config.json").write_text(json.dumps(model_config))

    args = SimpleNamespace(
        model_name_or_path=str(tmp_path),
        trust_remote_code=False,
        task=None,
        model_script=None,
        script_dir=None,
    )

    # execute - should not raise
    config = get_input_model_config(args)

    # model_path should remain unchanged since no onnx_file_name to guide rewriting
    assert config["config"]["model_path"] == stale_model_path


def _discrepancy_run_config():
    return {
        "input_model": {"type": "HfModel", "test_model_path": "ref_model"},
        "output_dir": "out_dir",
    }


def test_add_discrepancy_check_pass_default_enables_mae_only():
    from olive.cli.base import add_discrepancy_check_pass

    run_config = add_discrepancy_check_pass(_discrepancy_run_config())

    passes = run_config["passes"]
    # SaveTestModelConfig must be the first pass
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"

    pass_config = passes["discrepancy_check"]
    assert pass_config["type"] == "OnnxDiscrepancyCheck"
    assert pass_config["reference_model_path"] == str(Path("ref_model").resolve())
    # default: mae only -> test_metrics stores the human-readable selection
    assert pass_config["test_metrics"] == ["mae"]


def test_add_discrepancy_check_pass_speedup_only_disables_mae():
    from olive.cli.base import add_discrepancy_check_pass

    run_config = add_discrepancy_check_pass(_discrepancy_run_config(), metrics=["speedup"])

    passes = run_config["passes"]
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"

    pass_config = passes["discrepancy_check"]
    assert pass_config["test_metrics"] == ["speedup"]


def test_add_discrepancy_check_pass_mae_only_disables_speedup():
    from olive.cli.base import add_discrepancy_check_pass

    run_config = add_discrepancy_check_pass(_discrepancy_run_config(), metrics=["mae"])

    passes = run_config["passes"]
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"

    pass_config = passes["discrepancy_check"]
    assert pass_config["test_metrics"] == ["mae"]


def test_warn_unused_test_metrics_logs_when_test_disabled():
    from olive.cli.base import warn_unused_test_metrics

    with patch("olive.cli.base.logger") as mock_logger:
        warn_unused_test_metrics(test=None, metrics=["speedup"])

    mock_logger.warning.assert_called_once()
    assert "--test_metrics is ignored" in mock_logger.warning.call_args[0][0]


def test_warn_unused_test_metrics_silent_when_test_enabled():
    from olive.cli.base import warn_unused_test_metrics

    with patch("olive.cli.base.logger") as mock_logger:
        warn_unused_test_metrics(test=True, metrics=["speedup"])

    mock_logger.warning.assert_not_called()


def test_warn_unused_test_metrics_logs_llama_path_when_test_disabled():
    from olive.cli.base import warn_unused_test_metrics

    with patch("olive.cli.base.logger") as mock_logger:
        warn_unused_test_metrics(test=None, metrics=None, llama_path="/path/to/llama_env")

    mock_logger.warning.assert_called_once()
    assert "--test_llama_path is ignored" in mock_logger.warning.call_args[0][0]


def test_add_discrepancy_check_pass_llama_env_path_sets_config():
    from olive.cli.base import add_discrepancy_check_pass

    run_config = add_discrepancy_check_pass(_discrepancy_run_config(), llama_env_path="/path/to/llama_env")

    passes = run_config["passes"]
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"
    assert passes["convert_hf_to_gguf"]["type"] == "ConvertHfToGGUF"
    assert passes["convert_hf_to_gguf"]["llama_cpp_env_path"] == "/path/to/llama_env"

    pass_config = passes["discrepancy_check"]
    assert pass_config["llama_cpp"] is True
    assert pass_config["llama_cpp_env_path"] == "/path/to/llama_env"


def test_add_discrepancy_check_pass_no_llama_env_path_omits_llama_config():
    from olive.cli.base import add_discrepancy_check_pass

    run_config = add_discrepancy_check_pass(_discrepancy_run_config())

    passes = run_config["passes"]
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"

    pass_config = passes["discrepancy_check"]
    assert "convert_hf_to_gguf" not in passes
    assert "llama_cpp" not in pass_config
    assert "llama_cpp_env_path" not in pass_config


def test_add_discrepancy_check_pass_updates_existing_pass():
    """When OnnxDiscrepancyCheck already exists in the config, its runtime fields are updated."""
    from olive.cli.base import add_discrepancy_check_pass

    # Simulate a config generated by `olive optimize --dry_run --test` - the pass already exists
    # with stale settings (old output dir, only mae was requested at generate-time).
    config = _discrepancy_run_config()
    config["passes"] = {
        "discrepancy_check": {
            "type": "OnnxDiscrepancyCheck",
            "reference_model_path": "/old/abs/path",
            "report_output_dir": "/old/out_dir",
            "test_metrics": ["mae"],
        }
    }
    config["input_model"]["test_model_path"] = "new_ref_model"
    config["output_dir"] = "new_out_dir"

    result = add_discrepancy_check_pass(config, metrics=["mae", "speedup"], llama_env_path="/path/to/llama_env")

    passes = result["passes"]
    # SaveTestModelConfig must be injected at the beginning
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"
    assert passes["convert_hf_to_gguf"]["type"] == "ConvertHfToGGUF"
    assert passes["convert_hf_to_gguf"]["llama_cpp_env_path"] == "/path/to/llama_env"
    assert passes["convert_hf_to_gguf"]["reference_model_path"] == str(Path("new_ref_model").resolve())

    pass_config = passes["discrepancy_check"]
    # Reference model path and output dir must be updated to the current values.
    assert pass_config["reference_model_path"] == str(Path("new_ref_model").resolve())
    assert pass_config["report_output_dir"] == "new_out_dir"
    # test_metrics must reflect the newly requested metrics.
    assert pass_config["test_metrics"] == ["mae", "speedup"]


def test_add_discrepancy_check_pass_updates_existing_pass_speedup_only():
    """Updating an existing pass with speedup-only metrics updates test_metrics."""
    from olive.cli.base import add_discrepancy_check_pass

    config = _discrepancy_run_config()
    config["passes"] = {
        "dc": {
            "type": "onnxdiscrepancycheck",  # case-insensitive type match
            "reference_model_path": "/old/path",
            "test_metrics": ["mae"],
        }
    }

    result = add_discrepancy_check_pass(config, metrics=["speedup"])

    passes = result["passes"]
    # SaveTestModelConfig must be injected at the beginning
    first_key = next(iter(passes))
    assert passes[first_key]["type"] == "SaveTestModelConfig"

    pass_config = passes["dc"]
    assert pass_config["test_metrics"] == ["speedup"]

    from olive.cli.base import _parse_test_metrics

    assert _parse_test_metrics("mae,speedup") == ["mae", "speedup"]


def test_parse_test_metrics_single():
    from olive.cli.base import _parse_test_metrics

    assert _parse_test_metrics("mae") == ["mae"]


def test_parse_test_metrics_accepts_generation_metrics():
    from olive.cli.base import _parse_test_metrics

    assert _parse_test_metrics("first_token_20,tft,tf5t") == ["first_token_20", "tft", "tf5t"]


def test_parse_test_metrics_invalid_raises():
    import argparse

    from olive.cli.base import _parse_test_metrics

    with pytest.raises(argparse.ArgumentTypeError, match="invalid choice"):
        _parse_test_metrics("unknown")


def test_flatten_test_metrics_nested_lists():
    from olive.cli.base import _flatten_test_metrics

    # Simulates: --test_metrics mae,speedup  → [["mae", "speedup"]]
    assert _flatten_test_metrics([["mae", "speedup"]]) == ["mae", "speedup"]


def test_flatten_test_metrics_space_separated_tokens():
    from olive.cli.base import _flatten_test_metrics

    # Simulates: --test_metrics mae speedup  → [["mae"], ["speedup"]]
    assert _flatten_test_metrics([["mae"], ["speedup"]]) == ["mae", "speedup"]


def test_flatten_test_metrics_none_returns_none():
    from olive.cli.base import _flatten_test_metrics

    assert _flatten_test_metrics(None) is None
