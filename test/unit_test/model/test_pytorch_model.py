# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import unittest
from types import FunctionType
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
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
        self.architecture = "google/bert_uncased_L-4_H-256_A-4"
        self.original_model = transformers.BertForSequenceClassification.from_pretrained(self.architecture)
        # note that cannot tokenizer cannot be used before any forked process
        # otherwise it will disable parallelism to avoid deadlocks which make the
        # pytest.mark.parametrize unavailable
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.architecture)
        self.input_text = "Today was an amazing day."
        self.hf_conf = {
            # this is `task_type` and not `task` since it's an input to aml_mlflow.hftransformers.save_model
            # which expects `task_type` as the key
            "task_type": self.task,
        }

        # cleanup the model path, otherwise, the test will fail after the first run.
        aml_mlflow.hftransformers.save_model(
            self.original_model,
            self.model_path,
            tokenizer=self.tokenizer,
            config=self.original_model.config,
            hf_conf=self.hf_conf,
        )
        yield
        shutil.rmtree(self.root_dir, ignore_errors=True)

    def test_mlflow_model_hfconfig_function(self):
        hf_model = PyTorchModelHandler(
            model_path=self.architecture, hf_config={"task": self.task, "model_name": self.architecture}
        )
        mlflow_olive_model = PyTorchModelHandler(
            model_path=self.model_path,
            model_file_format="PyTorch.MLflow",
            hf_config={
                "task": self.task,
                "model_name": self.architecture,
            },
        )
        # load_hf_model only works for huggingface models and not for mlflow models
        assert mlflow_olive_model.get_hf_model_config() == hf_model.get_hf_model_config()
        assert mlflow_olive_model.get_hf_io_config() == hf_model.get_hf_io_config()
        assert len(list(mlflow_olive_model.get_hf_components())) == len(list(hf_model.get_hf_components()))
        assert len(mlflow_olive_model.get_hf_dummy_inputs()) == len(hf_model.get_hf_dummy_inputs())

    def test_hf_model_attributes(self):
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.architecture})
        # model_attributes will be delayed loaded until pass run
        assert olive_model.model_attributes == transformers.AutoConfig.from_pretrained(self.architecture).to_dict()

    def test_load_model_with_pretrained_args(self):
        # as the tokenizer is used in setup function
        # we cannot use pytest.mark.parametrize as it will be disabled
        # to avoid deadlocks
        olive_model = PyTorchModelHandler(
            model_path=self.model_path,
            model_file_format="PyTorch.MLflow",
            hf_config={
                "task": self.task,
                "model_name": self.architecture,
                "from_pretrained_args": {"trust_remote_code": True},
            },
        ).load_model()
        assert olive_model is not None

    def test_load_model(self):
        olive_model = PyTorchModelHandler(model_path=self.model_path, model_file_format="PyTorch.MLflow").load_model()
        mlflow_model = mlflow.pyfunc.load_model(self.model_path)

        sample_input = {"inputs": {"input_string": self.input_text}}
        mlflow_predict_result = mlflow_model.predict(pd.DataFrame.from_dict(sample_input)).values[0]  # noqa: PD011

        encoded_input = self.tokenizer(
            self.input_text,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt",
        )
        olive_result = (
            olive_model(input_ids=encoded_input["input_ids"], attention_mask=encoded_input["attention_mask"])
            .logits.argmax()
            .item()
        )
        olive_predict_result = [olive_model.config.id2label[olive_result]]

        assert mlflow_predict_result == olive_predict_result


class TestPyTorchHFModel(unittest.TestCase):
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "BertForSequenceClassification"
        self.model_name = "Intel/bert-base-uncased-mrpc"
        self.torch_dtype = "float16"

    def test_hf_config_task(self):
        self.setup()

        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_config_model_class(self):
        self.setup()

        olive_model = PyTorchModelHandler(hf_config={"model_class": self.model_class, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_from_pretrained_args(self):
        self.setup()

        olive_model = PyTorchModelHandler(
            hf_config={
                "task": self.task,
                "model_name": self.model_name,
                "from_pretrained_args": {"torch_dtype": self.torch_dtype},
            }
        )
        pytorch_model = olive_model.load_model()
        assert pytorch_model.dtype == getattr(torch, self.torch_dtype)


class TestPytorchDummyInput:
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "BertForSequenceClassification"
        self.model_name = "Intel/bert-base-uncased-mrpc"
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
        self.dataset = {
            "data_name": "glue",
            "subset": "mrpc",
            "split": "validation",
            "input_cols": ["sentence1", "sentence2"],
            "label_cols": ["label"],
            "batch_size": 1,
        }

    def test_dummy_input_with_kv_cache(self):
        io_config = self.io_config
        io_config["kv_cache_config"] = True
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=io_config
        )
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 12 * 2
        assert list(dummy_inputs["past_key_0"].shape) == [1, 12, 0, 64]

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
        assert io_config == IoConfig(**self.io_config).dict()

    @patch("olive.model.handler.mixin.hf_config.get_hf_model_io_config")
    def test_hf_config_io_config(self, get_hf_model_io_config):
        get_hf_model_io_config.return_value = self.io_config
        olive_model = PyTorchModelHandler(hf_config={"task": self.task, "model_name": self.model_name})
        # get io config
        io_config = olive_model.io_config
        assert io_config == self.io_config
        get_hf_model_io_config.assert_called_once_with(self.model_name, self.task, None)

    def common_data_config_test(self, olive_model, data_config_template):
        # mock data config
        data_config = MagicMock()
        # mock data container
        data_container = MagicMock()
        data_container.get_first_batch.return_value = 1, 0
        data_config.to_data_container.return_value = data_container
        # mock data config template
        data_config_template.return_value = data_config

        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        data_config_template.assert_called_once()
        data_config.to_data_container.assert_called_once()
        data_container.get_first_batch.assert_called_once()
        assert dummy_inputs == 1

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
        self.common_data_config_test(olive_model, dummy_data_config_template)

    @patch("olive.data.template.huggingface_data_config_template")
    def test_hf_config_dataset_dummy_inputs(self, hf_data_config_template):
        olive_model = PyTorchModelHandler(
            hf_config={"task": self.task, "model_name": self.model_name, "dataset": self.dataset}
        )
        self.common_data_config_test(olive_model, hf_data_config_template)

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
