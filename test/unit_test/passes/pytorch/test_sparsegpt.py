# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.data.template import huggingface_data_config_template
from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import SparseGPT


def test_sparsegpt(tmp_path):
    # setup
    model_name = "sshleifer/tiny-gpt2"
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
                "corpus_strategy": "join-random",
                "source_max_len": 1024,
                "max_samples": 1,
                "random_seed": 42,
            }
        },
    }
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset)
    config = {
        "sparsity": [2, 4],
        "data_config": data_config,
    }

    p = create_pass_from_dict(SparseGPT, config, disable_search=True)
    output_folder = str(tmp_path / "sparse")

    # execute
    p.run(input_model, None, output_folder)
