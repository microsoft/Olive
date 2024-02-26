# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
import pytest
import torch

from olive.model.handler.pytorch import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import GptqQuantizer


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gptq requires GPU.",
)
def test_quantization_default(tmp_path: Path):
    # setup
    input_model = PyTorchModelHandler(
        hf_config={
            "model_class": "OPTForCausalLM",
            "model_name": "facebook/opt-125m",
        }
    )
    config = {}

    p = create_pass_from_dict(GptqQuantizer, config, disable_search=True)
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    p.run(input_model, None, gptq_out_folder)
