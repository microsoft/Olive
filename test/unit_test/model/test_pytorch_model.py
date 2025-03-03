# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import patch

import pytest

from olive.model.config.io_config import IoConfig
from olive.model.handler.pytorch import PyTorchModelHandler


@pytest.fixture(name="io_config")
def io_config_fixture():
    return {
        "input_names": ["input_ids", "attention_mask", "token_type_ids"],
        "input_shapes": [[1, 128], [1, 128], [1, 128]],
        "input_types": ["int64", "int64", "int64"],
        "output_names": ["output"],
        "dynamic_axes": {
            "input_ids": {"0": "batch_size", "1": "seq_length"},
            "attention_mask": {"0": "batch_size", "1": "seq_length"},
            "token_type_ids": {"0": "batch_size", "1": "seq_length"},
        },
        "dynamic_shapes": {
            "input_ids": {"0": "batch_size", "1": ["seq_length", 1, 256]},
            "attention_mask": {"0": "batch_size", "1": "seq_length"},
            "token_type_ids": {"0": "batch_size", "1": "seq_length"},
        },
    }


@pytest.fixture(name="model_scripts")
def model_scripts_fixture(tmp_path):
    script_dir = tmp_path / "model"
    script_dir.mkdir(exist_ok=True)
    model_script = script_dir / "model_script.py"
    model_script.write_text("def model(): pass")
    return {
        "script_dir": str(script_dir),
        "model_script": str(model_script),
    }


def test_model_to_json(model_scripts):
    model = PyTorchModelHandler(model_path="test_path", **model_scripts)
    model.set_resource("model_script", "model_script")
    model_json = model.to_json()
    assert model_json["config"]["model_path"] == "test_path"
    assert model_json["config"]["script_dir"] == model_scripts["script_dir"]
    assert model_json["config"]["model_script"] == "model_script"


@patch("torch.load")
def test_load_from_path(torch_load):
    torch_load.return_value = "dummy_pytorch_model"

    model = PyTorchModelHandler(model_path="test_path")

    assert model.load_model() == "dummy_pytorch_model"
    torch_load.assert_called_once_with("test_path", weights_only=False)


@patch("olive.model.handler.pytorch.UserModuleLoader")
def test_load_from_loader(user_module_loader, model_scripts):
    user_module_loader.return_value.call_object.return_value = "dummy_pytorch_model"

    model = PyTorchModelHandler(model_path="dummy_path", model_loader="dummy_loader", **model_scripts)

    assert model.load_model() == "dummy_pytorch_model"

    user_module_loader.assert_called_once_with(model_scripts["model_script"], model_scripts["script_dir"])
    user_module_loader.return_value.call_object.assert_called_once_with("dummy_loader", "dummy_path")


def test_io_config(io_config):
    olive_model = PyTorchModelHandler(model_path="dummy", io_config=io_config)
    assert olive_model.io_config == IoConfig(**io_config).dict(exclude_none=True)


@patch("olive.data.template.dummy_data_config_template")
def test_input_shapes_dummy_inputs(dummy_data_config_template, io_config):
    olive_model = PyTorchModelHandler(model_path="dummy", io_config=io_config)

    dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.return_value = 1, 0

    # get dummy inputs
    dummy_inputs = olive_model.get_dummy_inputs()

    dummy_data_config_template.assert_called_once_with(
        input_shapes=io_config["input_shapes"],
        input_types=io_config["input_types"],
        input_names=io_config["input_names"],
    )
    dummy_data_config_template.return_value.to_data_container.assert_called_once()
    dummy_data_config_template.return_value.to_data_container.return_value.get_first_batch.assert_called_once()

    assert dummy_inputs == 1
