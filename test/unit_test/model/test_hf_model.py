# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import unittest
from unittest.mock import MagicMock, patch

import pytest
import torch
import transformers
from azureml.evaluate import mlflow as aml_mlflow

from olive.common.utils import dict_diff
from olive.model.config.io_config import IoConfig
from olive.model.handler.hf import HfModelHandler


# pylint: disable=attribute-defined-outside-init
class TestHfMLflowModel(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        self.root_dir = tmp_path
        self.model_path = str(self.root_dir.resolve() / "mlflow_test")
        self.task = "text-classification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"
        # cache dir where the mlflow transformers model is saved
        original_cache_dir = os.environ.get("OLIVE_CACHE_DIR", None)
        os.environ["OLIVE_CACHE_DIR"] = str(self.root_dir / "cache")

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

        yield

        if original_cache_dir:
            os.environ["OLIVE_CACHE_DIR"] = original_cache_dir
        else:
            os.environ.pop("OLIVE_CACHE_DIR", None)

    def test_mlflow_model_hfconfig_function(self):
        hf_model = HfModelHandler(model_path=self.model_path, task=self.task)
        mlflow_olive_model = HfModelHandler(model_path=self.model_path, task=self.task)

        # load_hf_model only works for huggingface models and not for mlflow models
        assert mlflow_olive_model.get_hf_io_config() == hf_model.get_hf_io_config()
        assert len(mlflow_olive_model.get_hf_dummy_inputs()) == len(hf_model.get_hf_dummy_inputs())

    def test_hf_model_attributes(self):
        olive_model = HfModelHandler(model_path=self.model_path, task=self.task)
        original_hf_model_config = transformers.AutoConfig.from_pretrained(self.model_name).to_dict()
        # "_name_or_path" is expected to be different since it points to where the config
        # was loaded from
        assert olive_model.model_attributes.keys() == original_hf_model_config.keys()
        difference = dict_diff(olive_model.model_attributes, original_hf_model_config)
        assert len(difference) == 1
        assert "_name_or_path" in difference

    def test_load_model(self):
        olive_model = HfModelHandler(model_path=self.model_path, task=self.task).load_model()

        assert isinstance(olive_model, transformers.BertForSequenceClassification)

    def test_load_model_with_kwargs(self):
        # as the tokenizer is used in setup function
        # we cannot use pytest.mark.parametrize as it will be disabled
        # to avoid deadlocks
        olive_model = HfModelHandler(
            model_path=self.model_path, task=self.task, load_kwargs={"torch_dtype": "float16"}
        ).load_model()
        assert isinstance(olive_model, transformers.BertForSequenceClassification)
        assert olive_model.dtype == torch.float16

    def test_model_name_or_path(self):
        olive_model = HfModelHandler(model_path=self.model_path, task=self.task)
        assert olive_model.model_name_or_path.startswith(os.environ["OLIVE_CACHE_DIR"])


class TestHFModel(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    def test_load_model(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)

    def test_load_model_with_kwargs(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, load_kwargs={"torch_dtype": "float16"})
        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.BertForSequenceClassification)
        assert pytorch_model.dtype == torch.float16

    def test_model_name_or_path(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        assert olive_model.model_name_or_path == self.model_name


class TestHFDummyInput(unittest.TestCase):
    @pytest.fixture(autouse=True)
    def setup(self):
        # hf config values
        self.task = "text-classification"
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
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dummy_input_with_kv_cache_dict(self):
        io_config = self.io_config
        io_config["kv_cache"] = {"batch_size": 1}
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=io_config)
        dummy_inputs = olive_model.get_dummy_inputs()
        # len(["input_ids", "attention_mask", "token_type_ids"]) + 2 * num_hidden_layers
        assert len(dummy_inputs) == 3 + 5 * 2
        assert list(dummy_inputs["past_key_values.0.key"].shape) == [1, 4, 0, 8]

    def test_dict_io_config(self):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)
        # get io config
        io_config = olive_model.io_config
        assert io_config == IoConfig(**self.io_config).dict(exclude_none=True)

    @patch("olive.model.handler.mixin.hf.get_model_io_config")
    def test_hf_config_io_config(self, get_model_io_config):
        get_model_io_config.return_value = self.io_config
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get io config
        io_config = olive_model.io_config
        assert io_config == self.io_config
        get_model_io_config.assert_called_once_with(self.model_name, self.task)

    @patch("olive.data.template.dummy_data_config_template")
    def test_input_shapes_dummy_inputs(self, dummy_data_config_template):
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task, io_config=self.io_config)
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

    @patch("olive.model.handler.mixin.hf.get_model_dummy_input")
    def test_hf_onnx_config_dummy_inputs(self, get_model_dummy_input):
        get_model_dummy_input.return_value = 1
        olive_model = HfModelHandler(model_path=self.model_name, task=self.task)
        # get dummy inputs
        dummy_inputs = olive_model.get_dummy_inputs()

        get_model_dummy_input.assert_called_once_with(self.model_name, self.task)
        assert dummy_inputs == 1
