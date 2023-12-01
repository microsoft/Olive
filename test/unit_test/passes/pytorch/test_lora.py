# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
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


def get_pass_config(model_name, task, **kwargs):
    dataset = {
        "data_name": "ptb_text_only",
        "subset": "penn_treebank",
        "split": "train",
        "component_kwargs": {
            "pre_process_data": {
                "text_cols": ["sentence"],
                "corpus_strategy": "line-by-line",
                "source_max_len": 512,
                "max_samples": 10,
                "pad_to_max_len": False,
            }
        },
    }
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
        **kwargs,
    }


def run_finetuning(pass_class, tmp_path, **pass_config_kwargs):
    # setup
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    input_model = PyTorchModel(hf_config={"model_name": model_name, "task": task})
    # convert to json to ensure the pass can handle serialized data config
    config = get_pass_config(model_name, task, **pass_config_kwargs)
    p = create_pass_from_dict(pass_class, config, disable_search=True)
    output_folder = str(tmp_path / "output_model")

    # execute
    return p.run(input_model, None, output_folder)


def test_lora(tmp_path):
    # execute
    out = run_finetuning(LoRA, tmp_path)

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
    # execute
    out = run_finetuning(QLoRA, tmp_path)

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.fixture(name="mock_torch_ort")
def mock_torch_ort_fixture():
    # mock torch_ort since we don't install it in the test environment
    mock_torch_ort = MagicMock()
    sys.modules["torch_ort"] = mock_torch_ort
    yield mock_torch_ort
    del sys.modules["torch_ort"]


@pytest.fixture(name="clean_env")
def clean_env_fixture():
    yield
    if "ORTMODULE_ONNX_OPSET_VERSION" in os.environ:
        del os.environ["ORTMODULE_ONNX_OPSET_VERSION"]


@pytest.mark.usefixtures("clean_env", "mock_torch_ort")
@pytest.mark.parametrize("value,expected_value", [(None, "16"), (-1, None), (16, "16"), (15, "15"), (17, "17")])
@patch("olive.passes.pytorch.lora.LoRA.train_and_save_new_model")
@patch("optimum.onnxruntime.utils.is_onnxruntime_training_available", return_value=True)
@patch("onnxruntime.__version__", "1.17.0")
def test_ortmodule_onnx_opset_version(_, tmp_path, caplog, value, expected_value):
    # execute
    pass_config_kwargs = {"use_ort_trainer": True}
    if value is not None:
        pass_config_kwargs["ortmodule_onnx_opset_version"] = value

    if expected_value is None:
        # invalid value
        with pytest.raises(AssertionError):
            run_finetuning(LoRA, tmp_path, **pass_config_kwargs)
    else:
        # capture logging to check for opset < 16 warning
        logger = logging.getLogger("olive")
        logger.propagate = True

        run_finetuning(LoRA, tmp_path, **pass_config_kwargs)

        # assert
        assert os.environ["ORTMODULE_ONNX_OPSET_VERSION"] == expected_value
        if int(expected_value) < 16:
            assert "training with bfloat16 might not work properly" in caplog.text
