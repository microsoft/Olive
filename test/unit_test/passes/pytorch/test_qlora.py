# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from unittest.mock import patch

from olive.data.template import huggingface_data_config_template
from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import QLoRA


def patched_find_all_linear_names(model):
    return ["k_proj", "v_proj", "out_proj", "q_proj", "fc1", "fc2"]


# quantization requires gpu so we will patch the model loading args with no quantization
@patch("olive.passes.pytorch.qlora.HFModelLoadingArgs")
@patch("olive.passes.pytorch.qlora.QLoRA.find_all_linear_names", side_effect=patched_find_all_linear_names)
def test_qlora(patched_model_loading_args, patched_find_all_linear_names, tmp_path):
    # setup
    model_name = "hf-internal-testing/tiny-random-OPTForCausalLM"
    task = "text-generation"
    input_model = PyTorchModel(hf_config={"model_name": model_name, "task": task})
    dataset = {
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
    # convert to json to ensure the pass can handle serialized data config
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset).to_json()
    config = {
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

    p = create_pass_from_dict(QLoRA, config, disable_search=True)
    output_folder = str(tmp_path / "qlora")

    # execute
    out = p.run(input_model, None, output_folder)
    assert Path(out.get_resource("adapter_path")).exists()
