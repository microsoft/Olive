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
                    "attn_implementation": "eager",
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
                    "attn_implementation": "eager",
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
                    "attn_implementation": "eager",
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
                    "attn_implementation": "eager",
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
    with pytest.raises(ValueError, match="_model_loader function is required in model_script for PyTorch model."):
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
    with pytest.raises(ValueError, match="invalid-name is not a valid Huggingface model name."):
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
