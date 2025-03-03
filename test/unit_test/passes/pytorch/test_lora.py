# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from pathlib import Path

import pytest
import torch

from olive.common.constants import OS
from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.lora import DoRA, LoftQ, LoHa, LoKr, LoRA, QLoRA

# pylint: disable=redefined-outer-name


def get_pass_config(model_name, task, **kwargs):
    dataset = {
        "load_dataset_config": {
            "params": {
                "data_name": "ptb_text_only",
                "subset": "penn_treebank",
                "split": "train",
                "trust_remote_code": True,
            }
        },
        "pre_process_data_config": {
            "params": {
                "text_cols": ["sentence"],
                "strategy": "line-by-line",
                "max_seq_len": 512,
                "max_samples": 10,
                "pad_to_max_len": False,
                "trust_remote_code": True,
            }
        },
    }
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset).to_json()
    return {
        "train_data_config": data_config,
        # hidden sizes are 4 or 16
        # will have invalid adapter weights since `in_features` and/or `out_features` say 64 (r) even though
        # the actual weights are 4 or 16. Bug not from our code, it's from peft
        "r": 4,
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
    input_model = HfModelHandler(model_path=model_name, task=task)
    # convert to json to ensure the pass can handle serialized data config
    config = get_pass_config(model_name, task, **pass_config_kwargs)
    p = create_pass_from_dict(pass_class, config, disable_search=True)
    output_folder = str(tmp_path / "output_model")

    # execute
    return p.run(input_model, output_folder)


# TODO(team): Failed in pipeline (linux gpu). Need to investigate.
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


# TODO(team): Failed in pipeline (linux gpu). Need to investigate.
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available() or True,
    reason="bitsandbytes requires Linux GPU.",
)
def test_qlora(tmp_path):
    # execute
    # bfloat16 is not supported on all gpu
    out = run_finetuning(QLoRA, tmp_path, torch_dtype="float32", device_map="current_device")

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


# TODO(team): Failed in pipeline (linux gpu). Need to investigate.
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available() or True,
    reason="bitsandbytes requires Linux GPU.",
)
def test_loftq(tmp_path):
    # execute
    # bfloat16 is not supported on all gpu
    out = run_finetuning(LoftQ, tmp_path, torch_dtype="float32", device_map="current_device")

    # assert
    assert Path(out.get_resource("model_path")).exists()
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available(),
    reason="bitsandbytes requires Linux GPU.",
)
def test_loha(tmp_path):
    # execute
    out = run_finetuning(
        LoHa, tmp_path, torch_dtype="float16", training_args={"remove_unused_columns": False, "save_safetensors": False}
    )

    # assert
    assert Path(out.get_resource("adapter_path")).exists()


@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available(),
    reason="bitsandbytes requires Linux GPU.",
)
def test_lokr(tmp_path):
    # execute
    out = run_finetuning(
        LoKr, tmp_path, torch_dtype="float16", training_args={"remove_unused_columns": False, "save_safetensors": False}
    )

    assert Path(out.get_resource("adapter_path")).exists()


@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or not torch.cuda.is_available(),
    reason="bitsandbytes requires Linux GPU.",
)
def test_dora(tmp_path):
    # execute
    out = run_finetuning(DoRA, tmp_path, torch_dtype="float32")

    # assert
    assert Path(out.get_resource("adapter_path")).exists()
