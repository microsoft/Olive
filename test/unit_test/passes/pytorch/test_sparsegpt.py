# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.sparsegpt import SparseGPT


def test_sparsegpt(tmp_path):
    # setup
    model_name = "sshleifer/tiny-gpt2"
    task = "text-generation"
    input_model = HfModelHandler(model_path=model_name, task=task)
    component_configs = {
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
                "strategy": "join-random",
                "max_seq_len": 1024,
                "max_samples": 1,
                "random_seed": 42,
                "trust_remote_code": True,
            }
        },
    }
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **component_configs)
    config = {
        "sparsity": [2, 4],
        "data_config": data_config,
    }

    p = create_pass_from_dict(SparseGPT, config, disable_search=True)
    output_folder = str(tmp_path / "sparse")

    # execute
    p.run(input_model, output_folder)
