# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.sparsegpt import SparseGPT
from test.utils import get_wikitext_data_config


def test_sparsegpt(tmp_path):
    # setup
    model_name = "sshleifer/tiny-gpt2"
    task = "text-generation"
    input_model = HfModelHandler(model_path=model_name, task=task)
    config = {"sparsity": [2, 4], "data_config": get_wikitext_data_config(model_name)}

    p = create_pass_from_dict(SparseGPT, config, disable_search=True)
    output_folder = str(tmp_path / "sparse")

    # execute
    p.run(input_model, output_folder)
