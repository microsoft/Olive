from types import SimpleNamespace
from unittest.mock import patch

import pytest

from olive.cli.base import get_model_name_or_path, get_pt_model_path, insert_input_model


@pytest.mark.parametrize(
    ("model_name_or_path", "expected_output"),
    [
        (
            "azureml://registries/my_registry/models/my_model/versions/1",
            {
                "type": "azureml_registry_model",
                "registry_name": "my_registry",
                "name": "my_model",
                "version": "1",
            },
        ),
        (
            "https://huggingface.co/my_model/my_model",
            "my_model/my_model",
        ),
        (
            "my_model",
            "my_model",
        ),
    ],
)
def test_get_model_name_or_path(model_name_or_path, expected_output):
    assert get_model_name_or_path(model_name_or_path) == expected_output


@pytest.mark.parametrize(
    ("model_name_or_path", "expected_output"),
    [
        (
            "azureml:my_model:1",
            {
                "type": "azureml_model",
                "name": "my_model",
                "version": "1",
            },
        ),
        (
            "my_model",
            "my_model",
        ),
    ],
)
def test_get_pt_model_path(model_name_or_path, expected_output):
    assert get_pt_model_path(model_name_or_path) == expected_output


def test_insert_input_model_hf_model():
    # setup
    config = {}
    args = SimpleNamespace(
        model_name_or_path="https://huggingface.co/my_model/my_model",
        trust_remote_code=True,
        task="classification",
        model_script=None,
        script_dir=None,
    )

    # execute
    insert_input_model(config, args)

    expected_config = {
        "input_model": {
            "type": "HfModel",
            "model_path": "my_model/my_model",
            "load_kwargs": {"trust_remote_code": True},
            "task": "classification",
        }
    }

    # assert
    assert config == expected_config


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
        (
            "azureml:my_model:1",  # model_name_or_path
            False,  # trust_remote_code
            None,  # task
            "model.py",  # model_script
            "scripts",  # script_dir
            {"_model_loader": False, "_io_config": True, "_dummy_inputs": False},  # has_function results
            {  # expected config
                "input_model": {
                    "type": "PyTorchModel",
                    "model_script": "model.py",
                    "script_dir": "scripts",
                    "io_config": "_io_config",
                    "model_path": {"type": "azureml_model", "name": "my_model", "version": "1"},
                }
            },
        ),
        (
            None,  # model_name_or_path
            False,  # trust_remote_code
            None,  # task
            "model.py",  # model_script
            "scripts",  # script_dir
            {"_model_loader": True, "_io_config": False, "_dummy_inputs": True},  # has_function results
            {  # expected config
                "input_model": {
                    "type": "PyTorchModel",
                    "model_script": "model.py",
                    "script_dir": "scripts",
                    "model_loader": "_model_loader",
                    "dummy_inputs_func": "_dummy_inputs",
                }
            },
        ),
    ],
)
@patch("olive.cli.base.UserModuleLoader")
def test_insert_input_model_pt_model(
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

    def has_function_side_effect(arg):
        return has_function_results.get(arg, False)

    mock_instance.has_function.side_effect = has_function_side_effect

    # execute
    insert_input_model(config, args)

    # assert
    assert config == expected_config


@patch("olive.cli.base.UserModuleLoader")
def test_insert_input_model_pt_model_missing_loader(MockUserModuleLoader):
    # setup
    config = {}
    args = SimpleNamespace(
        model_name_or_path=None,
        trust_remote_code=False,
        task=None,
        model_script="model.py",
        script_dir="scripts",
    )
    MockUserModuleLoader.return_value.has_function.return_value = False

    # execute and assert
    with pytest.raises(ValueError, match="_model_loader is required for PyTorch model in the script"):
        insert_input_model(config, args)
