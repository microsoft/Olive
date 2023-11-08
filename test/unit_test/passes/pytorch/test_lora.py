# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.data.template import huggingface_data_config_template
from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import LoRA, QLoRA

# pylint: disable=redefined-outer-name


def patched_find_submodules(*args, **kwargs):
    return ["k_proj", "v_proj", "out_proj", "q_proj", "fc1", "fc2"]


def get_dataset():
    return {
        "data_name": "ptb_text_only",
        "subset": "penn_treebank",
        "split": "train",
        "component_kwargs": {
            "pre_process_data": {
                "dataset_type": "corpus",
                "text_cols": ["sentence"],
                "corpus_strategy": "line-by-line",
                "source_max_len": 512,
                "max_samples": 10,
                "pad_to_max_len": False,
            }
        },
    }


def get_pass_config(model_name, task, **dataset):
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset).to_json()
    return {
        "train_data_config": data_config,
        "eval_dataset_size": 2,
        "training_args": {
            "per_device_train_batch_size": 1,
            "per_device_eval_batch_size": 1,
            "gradient_checkpointing": False,
            "max_steps": 2,
            "logging_steps": 1,
            "optim": "adamw_hf",
        },
    }


def test_lora(tmp_path):
    # setup
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    input_model = PyTorchModel(hf_config={"model_name": model_name, "task": task})
    dataset = get_dataset()
    # convert to json to ensure the pass can handle serialized data config
    config = get_pass_config(model_name, task, **dataset)
    p = create_pass_from_dict(LoRA, config, disable_search=True)
    output_folder = str(tmp_path / "lora")

    # execute
    out = p.run(input_model, None, output_folder)

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.fixture(name="mock_bitsandbytes")
def mock_bitsandbytes_fixture():
    # mock bitesandbtes since we don't install it in the test environment
    # it requires gpu and windows package is not published yet
    mock_bitsandbytes = MagicMock()
    sys.modules["bitsandbytes"] = mock_bitsandbytes
    yield mock_bitsandbytes
    del sys.modules["bitsandbytes"]


# quantization requires gpu so we will patch the model loading args with no quantization
@patch("olive.passes.pytorch.lora.HFModelLoadingArgs")
@patch("olive.passes.pytorch.lora.find_submodules", side_effect=patched_find_submodules)
def test_qlora(patched_model_loading_args, patched_find_submodules, tmp_path, mock_bitsandbytes):
    # setup
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    input_model = PyTorchModel(hf_config={"model_name": model_name, "task": task})
    dataset = get_dataset()
    # convert to json to ensure the pass can handle serialized data config
    config = get_pass_config(model_name, task, **dataset)
    p = create_pass_from_dict(QLoRA, config, disable_search=True)
    output_folder = str(tmp_path / "qlora")

    # execute
    out = p.run(input_model, None, output_folder)

    # assert
    assert Path(out.get_resource("adapter_path")).exists()
