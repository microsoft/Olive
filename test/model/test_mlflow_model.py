# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest
import torch
import transformers
import yaml

from olive.common.utils import dict_diff
from olive.model.handler.hf import HfModelHandler
from test.utils import get_tiny_phi3


@pytest.fixture(params=["aml_mlflow", "mlflow"], name="setup")
def setup_model(request, tmp_path):
    mode = request.param

    root_dir = tmp_path
    model_path = root_dir.resolve() / "mlflow_test"
    model_path.mkdir(parents=True, exist_ok=True)

    # Cache dir where the MLflow transformers model is saved
    original_cache_dir = os.environ.get("OLIVE_CACHE_DIR", None)
    os.environ["OLIVE_CACHE_DIR"] = str(root_dir / "cache")

    model = get_tiny_phi3()

    if mode == "mlflow":
        mlmodel = {
            "flavors": {
                "transformers": {
                    "code": None,
                    "components": ["tokenizer"],
                    "framework": "pt",
                    "instance_type": "TextGenerationPipeline",
                    "model_binary": "model",
                    "pipeline_model_type": "Phi3ForCausalLM",
                    "task": "text-generation",
                    "tokenizer_type": "LlamaTokenizerFast",
                }
            }
        }
        model_dir = model_path / "model"
        config_dir = model_path / "model"
        tokenizer_dir = model_path / "components" / "tokenizer"
    else:
        mlmodel = {
            "flavors": {
                "hftransformersv2": {
                    "code": None,
                    "hf_config_class": "AutoConfig",
                    "hf_pretrained_class": "AutoModelForCausalLM",
                    "hf_tokenizer_class": "AutoTokenizer",
                    "model_data": "data",
                    "task_type": "text-generation",
                }
            }
        }
        model_dir = model_path / "data" / "model"
        config_dir = model_path / "data" / "config"
        tokenizer_dir = model_path / "data" / "tokenizer"

    with (model_path / "MLmodel").open("w") as m:
        yaml.dump(mlmodel, m)

    model.load_model().save_pretrained(model_dir)
    model.get_hf_model_config().save_pretrained(config_dir)
    model.get_hf_tokenizer().save_pretrained(tokenizer_dir)

    yield model_path, model

    # Cleanup: restore the original cache directory environment variable
    if original_cache_dir:
        os.environ["OLIVE_CACHE_DIR"] = original_cache_dir
    else:
        os.environ.pop("OLIVE_CACHE_DIR", None)


@pytest.mark.parametrize("load_kwargs", [None, {}, {"torch_dtype": torch.float16}])
def test_load_model(setup, load_kwargs):
    model_path, _ = setup

    olive_model = HfModelHandler(model_path=model_path, load_kwargs=load_kwargs).load_model()

    assert isinstance(olive_model, transformers.Phi3ForCausalLM)
    if load_kwargs and "torch_dtype" in load_kwargs:
        assert olive_model.dtype == torch.float16


def test_mlflow_model_hfconfig_function(setup):
    model_path, hf_model = setup

    mlflow_olive_model = HfModelHandler(model_path=model_path)

    # Ensure the MLflow model has the same IO config and dummy inputs as the HF model
    assert mlflow_olive_model.get_hf_io_config() == hf_model.get_hf_io_config()
    assert len(mlflow_olive_model.get_hf_dummy_inputs()) == len(hf_model.get_hf_dummy_inputs())


def test_hf_model_attributes(setup):
    model_path, hf_model = setup
    olive_model = HfModelHandler(model_path=model_path)
    original_hf_model_config = hf_model.get_hf_model_config().to_dict()

    # "_name_or_path" is expected to be different since it points to where the config was loaded from
    assert olive_model.model_attributes.keys() == original_hf_model_config.keys()
    difference = dict_diff(olive_model.model_attributes, original_hf_model_config)
    assert len(difference) == 1
    assert "_name_or_path" in difference


def test_model_name_or_path(setup):
    model_path, _ = setup
    olive_model = HfModelHandler(
        model_path=model_path,
    )
    assert olive_model.model_name_or_path.startswith(os.environ["OLIVE_CACHE_DIR"])
