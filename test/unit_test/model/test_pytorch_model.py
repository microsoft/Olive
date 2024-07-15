# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest
from types import FunctionType
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from azureml.evaluate import mlflow as aml_mlflow

from olive.model.config.io_config import IoConfig
from olive.model.handler.pytorch import PyTorchModelHandler

# pylint: disable=attribute-defined-outside-init


class TestPyTorchMLflowModel(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.root_dir = tmp_path
        self.model_path = str(self.root_dir.resolve() / "mlflow_test")
        self.task = "text-classification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

        original_model = transformers.BertForSequenceClassification.from_pretrained(self.model_name)
        tokenizer = transformers.AutoTokenizer.from_pretrained(self.model_name)
        aml_mlflow.hftransformers.save_model(
            original_model,
            self.model_path,
            tokenizer=tokenizer,
            config=original_model.config,
            hf_conf={
                "task_type": self.task,
            },
            pip_requirements=["transformers"],
        )

    def test_mlflow_model_hfconfig_function(self):
        hf_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})
        mlflow_olive_model = PyTorchModelHandler(
            model_path=self.model_path,
            model_file_format="PyTorch.MLflow",
            hf_config={"task": self.task},
        )
        # load_hf_model only works for huggingface models and not for mlflow models
        assert mlflow_olive_model.get_hf_io_config() == hf_model.get_hf_io_config()
        assert len(list(mlflow_olive_model.get_hf_components())) == len(list(hf_model.get_hf_components()))
        assert len(mlflow_olive_model.get_hf_dummy_inputs()) == len(hf_model.get_hf_dummy_inputs())

    def test_hf_model_attributes(self):
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})
        # model_attributes will be delayed loaded until pass run
        assert olive_model.model_attributes == transformers.AutoConfig.from_pretrained(self.model_name).to_dict()

    def test_load_model(self):
        olive_model = PyTorchModelHandler(model_path=self.model_path, model_file_format="PyTorch.MLflow").load_model()

        assert isinstance(olive_model, transformers.BertForSequenceClassification)

    def test_load_model_with_pretrained_args(self):
        # as the tokenizer is used in setup function
        # we cannot use pytest.mark.parametrize as it will be disabled
        # to avoid deadlocks
        olive_model = PyTorchModelHandler(
            model_path=self.model_path,
            model_file_format="PyTorch.MLflow",
            hf_config={
                "from_pretrained_args": {"torch_dtype": "float16"},
            },
        ).load_model()
        assert isinstance(olive_model, transformers.BertForSequenceClassification)
        assert olive_model.dtype == torch.float16


class TestPyTorchHFModel(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "BertForSequenceClassification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    def test_hf_config_task(self):
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_config_model_class(self):
        olive_model = PyTorchModelHandler(hf_config={"model_class": self.model_class, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_from_pretrained_args(self):
        olive_model = PyTorchModelHandler(
            hf_config={
                "task": self.task,
                "model_name": self.model_name,
                "from_pretrained_args": {"torch_dtype": "float16"},
            }
        )
        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)
        assert pytorch_model.dtype == torch.float16


class TestPytorchDummyInput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "BertForSequenceClassification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        self.io_config = {
            "input_names": ["input_ids", "attention_mask", "token_type_ids"],
            "input_shapes": [[1, 128], [1, 128], [1, 128]],
            "input_types": ["int64", "int64", "int64"],
            "output_names": ["output"],
            "dynamic_axes": {
                "input_ids": {"0": "batch_size", "1": "seq_length"},
                "attention_mask": {"0": "batch_size", "1": "seq_length"},
                "token_type_ids": {"0": "batch_size", "1": "seq_length"},
            },
        }

    def test_dummy_input_with_kv_cache(self):
        io_config = self.io_config
        io_config["kv_cache"] = True
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=io_config
        )
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dummy_input_with_kv_cache_dict(self):
        io_config = self.io_config
        io_config["kv_cache"] = {"batch_size": 1}
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=io_config
        )
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dict_io_config(self):
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=self.io_config
        )
        # get io config
        io_config = olive_model.io_config
        assert io_config == IoConfig(**self.io_config).dict(exclude_none=True)

    def test_func_io_config(self):
        io_config_func = MagicMock(spec=FunctionType)
        io_config_func.return_value = self.io_config
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=io_config_func
        )
        # get io config
        io_config = olive_model.io_config
        io_config_func.assert_called_once_with(olive_model)
        assert io_config == IoConfig(**self.io_config).dict(exclude_none=True)

    @patch("olive.model.handler.mixin.hf_config.get_hf_model_io_config")
    def test_hf_config_io_config(self, get_hf_model_io_config):
        get_hf_model_io_config.return_value = self.io_config
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})
        # get io config
        io_config = olive_model.io_config
        assert io_config == self.io_config
        get_hf_model_io_config.assert_called_once_with(self.model_name, self.task, None)

    def test_custom_dummy_inputs(self):
        dummy_inputs_func = MagicMock(spec=FunctionType)
        dummy_inputs_func.return_value = 1
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, dummy_inputs_func=dummy_inputs_func
        )
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        dummy_inputs_func.assert_called_once_with(olive_model)
        assert dummy_inputs == 1

    @patch("olive.data.template.dummy_data_config_template")
    def test_input_shapes_dummy_inputs(self, dummy_data_config_template):
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=self.io_config
        )
        # mock data config
        data_config = MagicMock()
        # mock data container
        data_container = MagicMock()
        data_container.get_first_batch.return_value = 1, 0
        data_config.to_data_container.return_value = data_container
        # mock data config template
        dummy_data_config_template.return_value = data_config

        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        dummy_data_config_template.assert_called_once()
        data_config.to_data_container.assert_called_once()
        data_container.get_first_batch.assert_called_once()
        assert dummy_inputs == 1

    @patch("olive.model.handler.mixin.hf_config.get_hf_model_dummy_input")
    def test_hf_onnx_config_dummy_inputs(self, get_hf_model_dummy_input):
        get_hf_model_dummy_input.return_value = 1
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        get_hf_model_dummy_input.assert_called_once_with(self.model_name, self.task, None)
        assert dummy_inputs == 1


class TestPyTorchModel:
    def test_model_to_json(self, tmp_path):
        script_dir = tmp_path / "model"
        script_dir.mkdir(exist_ok=True)
        model = PyTorchModelHandler(model_path="test_path", script_dir=script_dir)
        model.set_resource("model_script", "model_script")
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == "test_path"
        assert model_json["config"]["script_dir"] == str(script_dir)
        assert model_json["config"]["model_script"] == "model_script"
