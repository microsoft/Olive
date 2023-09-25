# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import tempfile
import unittest
from pathlib import Path
from types import FunctionType
from unittest.mock import MagicMock, patch

import mlflow
import pandas as pd
import pytest
import torch
import transformers
from azureml.evaluate import mlflow as aml_mlflow

from olive.model import PyTorchModel


class TestPyTorchMLflowModel(unittest.TestCase):
    def setup(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root_dir = Path(self.tempdir.name)
        self.model_path = str(self.root_dir.resolve() / "mlflow_test")
        self.task = "text-classification"
        self.architecture = "Intel/bert-base-uncased-mrpc"
        self.original_model = transformers.BertForSequenceClassification.from_pretrained(self.architecture)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(self.architecture)
        self.input_text = "Today was an amazing day."
        self.hf_conf = {
            "task_type": self.task,
        }

        # cleanup the model path, otherwise, the test will fail after the first run.
        shutil.rmtree(self.model_path, ignore_errors=True)
        aml_mlflow.hftransformers.save_model(
            self.original_model,
            self.model_path,
            tokenizer=self.tokenizer,
            config=self.original_model.config,
            hf_conf=self.hf_conf,
        )

    def test_hf_model_attributes(self):
        self.setup()

        olive_model = PyTorchModel(hf_config={"task": self.task, "model_name": self.architecture})
        # model_attributes will be delayed loaded until pass run
        assert olive_model.model_attributes == transformers.AutoConfig.from_pretrained(self.architecture).to_dict()

    def test_load_model(self):
        self.setup()

        olive_model = PyTorchModel(model_path=self.model_path, model_file_format="PyTorch.MLflow").load_model()
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
        self.tempdir.cleanup()


class TestPyTorchHFModel(unittest.TestCase):
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "BertForSequenceClassification"
        self.model_name = "Intel/bert-base-uncased-mrpc"
        self.torch_dtype = "float16"

    def test_hf_config_task(self):
        self.setup()

        olive_model = PyTorchModel(hf_config={"task": self.task, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_config_model_class(self):
        self.setup()

        olive_model = PyTorchModel(hf_config={"model_class": self.model_class, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_hf_model_loading_args(self):
        self.setup()

        olive_model = PyTorchModel(
            hf_config={
                "task": self.task,
                "model_name": self.model_name,
                "model_loading_args": {"torch_dtype": self.torch_dtype},
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
        olive_model = PyTorchModel(
            hf_config={"task": self.task, "model_name": self.model_name}, dummy_inputs_func=dummy_inputs_func
        )
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        dummy_inputs_func.assert_called_once_with(olive_model)
        assert dummy_inputs == 1

    @patch("olive.data.template.dummy_data_config_template")
    def test_input_shapes_dummy_inputs(self, dummy_data_config_template):
        olive_model = PyTorchModel(
            hf_config={"task": self.task, "model_name": self.model_name}, io_config=self.io_config
        )
        self.common_data_config_test(olive_model, dummy_data_config_template)

    @patch("olive.data.template.huggingface_data_config_template")
    def test_hf_config_dataset_dummy_inputs(self, hf_data_config_template):
        olive_model = PyTorchModel(
            hf_config={"task": self.task, "model_name": self.model_name, "dataset": self.dataset}
        )
        self.common_data_config_test(olive_model, hf_data_config_template)

    @patch("olive.model.get_hf_model_dummy_input")
    def test_hf_onnx_config_dummy_inputs(self, get_hf_model_dummy_input):
        get_hf_model_dummy_input.return_value = 1
        olive_model = PyTorchModel(hf_config={"task": self.task, "model_name": self.model_name})
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        get_hf_model_dummy_input.assert_called_once_with(self.model_name, self.task, None)
        assert dummy_inputs == 1


class TestPyTorchModel:
    def test_model_to_json(self, tmp_path):
        script_dir = tmp_path / "model"
        script_dir.mkdir(exist_ok=True)
        model = PyTorchModel(model_path="test_path", script_dir=script_dir)
        model.set_resource("model_script", "model_script")
        model_json = model.to_json()
        assert model_json["config"]["model_path"] == "test_path"
        assert model_json["config"]["script_dir"] == str(script_dir)
        assert model_json["config"]["model_script"] == "model_script"
