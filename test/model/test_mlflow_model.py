# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest
import torch
import transformers
from azureml.evaluate import mlflow as aml_mlflow

from olive.common.utils import dict_diff
from olive.model.handler.hf import HfModelHandler


@pytest.fixture(params=["aml_mlflow", "mlflow"], name="setup")
def setup_model(request, tmp_path):
    save_method = request.param
    root_dir = tmp_path
    model_path = str(root_dir.resolve() / "mlflow_test")
    task = "text-classification"
    model_name = "hf-internal-testing/tiny-random-BertForSequenceClassification"

    # Cache dir where the MLflow transformers model is saved
    original_cache_dir = os.environ.get("OLIVE_CACHE_DIR", None)
    os.environ["OLIVE_CACHE_DIR"] = str(root_dir / "cache")

    # Initialize model and tokenizer
    original_model = transformers.BertForSequenceClassification.from_pretrained(model_name)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)

    # Save model using the specified method (aml_mlflow or mlflow)
    if save_method == "aml_mlflow":
        aml_mlflow.hftransformers.save_model(
            original_model,
            model_path,
            tokenizer=tokenizer,
            config=original_model.config,
            hf_conf={
                "task_type": task,
            },
            pip_requirements=["transformers"],
        )
    elif save_method == "mlflow":
        from mlflow.transformers import save_model

        save_model(
            transformers_model={
                "model": original_model,
                "tokenizer": tokenizer,
            },
            path=model_path,
            task=task,
            pip_requirements=["transformers"],
        )

    yield model_path, task, model_name

    # Cleanup: restore the original cache directory environment variable
    if original_cache_dir:
        os.environ["OLIVE_CACHE_DIR"] = original_cache_dir
    else:
        os.environ.pop("OLIVE_CACHE_DIR", None)


@pytest.mark.parametrize("dtype", [None, "float16"])
def test_load_model_with_kwargs(setup, dtype):
    model_path, task, _ = setup
    load_kwargs = {"torch_dtype": dtype} if dtype else {}

    olive_model = HfModelHandler(model_path=model_path, task=task, load_kwargs=load_kwargs).load_model()

    assert isinstance(olive_model, transformers.BertForSequenceClassification)
    if dtype:
        assert olive_model.dtype == torch.float16


def test_mlflow_model_hfconfig_function(setup):
    model_path, task, model_name = setup

    hf_model = HfModelHandler(model_path=model_name, task=task)
    mlflow_olive_model = HfModelHandler(model_path=model_path, task=task)

    # Ensure the MLflow model has the same IO config and dummy inputs as the HF model
    assert mlflow_olive_model.get_hf_io_config() == hf_model.get_hf_io_config()
    assert len(mlflow_olive_model.get_hf_dummy_inputs()) == len(hf_model.get_hf_dummy_inputs())


def test_hf_model_attributes(setup):
    model_path, task, model_name = setup
    olive_model = HfModelHandler(model_path=model_path, task=task)
    original_hf_model_config = transformers.AutoConfig.from_pretrained(model_name).to_dict()

    # "_name_or_path" is expected to be different since it points to where the config was loaded from
    assert olive_model.model_attributes.keys() == original_hf_model_config.keys()
    difference = dict_diff(olive_model.model_attributes, original_hf_model_config)
    assert len(difference) == 1
    assert "_name_or_path" in difference


def test_load_model(setup):
    model_path, task, _ = setup
    olive_model = HfModelHandler(model_path=model_path, task=task).load_model()

    assert isinstance(olive_model, transformers.BertForSequenceClassification)


def test_model_name_or_path(setup):
    model_path, task, _ = setup
    olive_model = HfModelHandler(model_path=model_path, task=task)
    assert olive_model.model_name_or_path.startswith(os.environ["OLIVE_CACHE_DIR"])
