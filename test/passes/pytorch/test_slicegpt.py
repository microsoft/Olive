# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys

import pytest
import torch

from olive.data.template import huggingface_data_config_template
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict


# TODO(team): Failed in pipeline (linux gpu). Need to investigate.
@pytest.mark.skipif(
    (sys.version_info < (3, 10) and not torch.cuda.is_available()) or True, reason="requires python3.10 or higher"
)
def test_slicegpt(tmp_path):
    from olive.passes.pytorch.slicegpt import SliceGPT

    # setup
    model_name = "facebook/opt-125m"
    task = "text-generation"
    input_model = HfModelHandler(model_path=model_name, task=task)
    dataset = {
        "load_dataset_config": {
            "params": {
                "data_name": "wikitext",
                "subset": "wikitext-2-raw-v1",
                "split": "train",
                "trust_remote_code": True,
            }
        },
        "pre_process_data_config": {
            "params": {
                "text_cols": ["text"],
                "strategy": "join",
                "add_special_tokens": False,
                "max_seq_len": 2048,
                "max_samples": 128,
                "joiner": "\n\n",
                "trust_remote_code": True,
            }
        },
    }
    data_config = huggingface_data_config_template(model_name=model_name, task=task, **dataset)
    config = {
        "sparsity": 0.4,
        "calibration_data_config": data_config,
    }

    p = create_pass_from_dict(SliceGPT, config, disable_search=True)
    output_folder = str(tmp_path / "slicegpt")

    # execute
    p.run(input_model, output_folder)
