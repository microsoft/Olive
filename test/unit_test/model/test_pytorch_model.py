# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import tempfile
import unittest
from pathlib import Path

import mlflow
import pandas as pd
import transformers
from azureml.evaluate import mlflow as aml_mlflow

from olive.model import PyTorchModel


class TestPyTorchMLflowModel(unittest.TestCase):
    def setup(self):
        self.tempdir = Path(tempfile.TemporaryDirectory().name)
        self.model_path = str(self.tempdir.resolve() / "mlflow_test")
        self.task = "text-classification"
        self.architecture = "distilbert-base-uncased-finetuned-sst-2-english"
        self.original_model = transformers.AutoModelForSequenceClassification.from_pretrained(self.architecture)
        self.tokenizer = transformers.DistilBertTokenizerFast.from_pretrained(self.architecture)
        self.input_text = ["Today was an amazing day!"]
        self.hf_conf = {
            "task_type": self.task,
        }
        aml_mlflow.hftransformers.save_model(
            self.original_model,
            self.model_path,
            tokenizer=self.tokenizer,
            config=self.original_model.config,
            hf_conf=self.hf_conf,
        )

    def test_load_model(self):
        self.setup()

        olive_model = PyTorchModel(model_path=self.model_path, model_file_format="PyTorch.MLflow").load_model()
        mlflow_model = mlflow.pyfunc.load_model(self.model_path)

        sample_input = {"inputs": {"input_string": self.input_text}}
        mlflow_predict_result = mlflow_model.predict(pd.DataFrame.from_dict(sample_input)).values[0]

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
        shutil.rmtree(self.tempdir)


class TestPyTorchHFModel(unittest.TestCase):
    def setup(self):
        # hf config values
        self.task = "text-classification"
        self.model_class = "DistilBertForSequenceClassification"
        self.model_name = "distilbert-base-uncased-finetuned-sst-2-english"

    def test_hf_config_task(self):
        self.setup()

        olive_model = PyTorchModel(hf_config={"task": self.task, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.DistilBertForSequenceClassification)

    def test_hf_config_model_class(self):
        self.setup()

        olive_model = PyTorchModel(hf_config={"model_class": self.model_class, "model_name": self.model_name})

        pytorch_model = olive_model.load_model()
        assert isinstance(pytorch_model, transformers.DistilBertForSequenceClassification)
