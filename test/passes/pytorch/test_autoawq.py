# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, Phi3ForCausalLM

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.autoawq import AutoAWQQuantizer
from test.utils import get_tiny_phi3


# TODO(team): Fix autoawq compatibility with transformers>=4.57 (PytorchGELUTanh removed)
# https://github.com/hiyouga/LlamaFactory/issues/9247
@pytest.mark.skip(reason="autoawq incompatible with transformers>=4.57, need fix")
@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="awq requires GPU.",
)
def test_awq(tmp_path: Path):
    # setup
    tiny_phi3 = get_tiny_phi3()
    # autoawq requires sizes to be multiple of 64
    input_model_path = tmp_path / "input_model"
    config = tiny_phi3.get_hf_model_config()
    config.hidden_size = 64
    config.intermediate_size = 128
    AutoModelForCausalLM.from_config(config).save_pretrained(input_model_path)
    tiny_phi3.save_metadata(input_model_path)

    input_model = HfModelHandler(input_model_path)

    p = create_pass_from_dict(
        AutoAWQQuantizer,
        {"q_group_size": 32},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    awq_out_folder = str(tmp_path / "awq")

    # execute
    out = p.run(input_model, awq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)
    assert isinstance(out.load_model(), Phi3ForCausalLM)
