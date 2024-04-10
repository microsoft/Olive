# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys

import pytest

from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict


@pytest.mark.skipif(sys.version_info < (3, 10), reason="requires python3.10 or higher")
def test_slicegpt(tmp_path):
    from olive.passes.pytorch.slicegpt import SliceGPT

    # setup
    model_name = "facebook/opt-125m"
    task = "text-generation"
    input_model = PyTorchModelHandler(hf_config={"model_name": model_name, "task": task})
    config = {
        "sparsity": 0.4,
        "calibration_dataset": "wikitext2",
    }

    p = create_pass_from_dict(SliceGPT, config, disable_search=True)
    output_folder = str(tmp_path / "slicegpt")

    # execute
    p.run(input_model, None, output_folder)
