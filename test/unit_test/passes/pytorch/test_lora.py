# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import os
import platform
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch

from olive.common.constants import OS
from olive.data.template import huggingface_data_config_template
from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.lora import LoftQ, LoRA, QLoRA

# pylint: disable=redefined-outer-name


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
        # hidden sizes are 4 or 16
        # will have invalid adapter weights since `in_features` and/or `out_features` say 64 (lora_r) even though
        # the actual weights are 4 or 16. Bug not from our code, it's from peft
        "lora_r": 4,
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
    input_model = PyTorchModelHandler(hf_config={"model_name": model_name, "task": task})
    # convert to json to ensure the pass can handle serialized data config
    config = get_pass_config(model_name, task, **pass_config_kwargs)
    p = create_pass_from_dict(pass_class, config, disable_search=True)
    output_folder = str(tmp_path / "output_model")

    # execute
    return p.run(input_model, None, output_folder)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="lora finetuning requires GPU.",
)
def test_lora(tmp_path):
    # execute
    # bfloat16 is not supported on all gpu
    out = run_finetuning(LoRA, tmp_path, torch_dtype="float32")

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available(),
    reason="bitsandbytes requires Linux GPU.",
)
def test_qlora(tmp_path):
    # execute
    # bfloat16 is not supported on all gpu
    out = run_finetuning(QLoRA, tmp_path, torch_dtype="float32")

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available(),
    reason="bitsandbytes requires Linux GPU.",
)
def test_loftq(tmp_path):
    # execute
    # bfloat16 is not supported on all gpu
    out = run_finetuning(LoftQ, tmp_path, torch_dtype="float32")

    # assert
    assert Path(out.get_resource("model_path")).exists()
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
@pytest.mark.parametrize(("value", "expected_value"), [(None, "16"), (-1, None), (16, "16"), (15, "15"), (17, "17")])
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
