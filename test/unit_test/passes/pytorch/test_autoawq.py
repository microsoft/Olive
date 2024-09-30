# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.autoawq import AutoAWQQuantizer


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="awq requires GPU.",
)
def test_awq(tmp_path: Path):
    # setup
    input_model = HfModelHandler(model_path="facebook/opt-125m", load_kwargs={"use_safetensors": False})

    p = create_pass_from_dict(
        AutoAWQQuantizer,
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    awq_out_folder = str(tmp_path / "awq")

    # execute
    out = p.run(input_model, awq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)

    from transformers import OPTForCausalLM

    assert isinstance(out.load_model(), OPTForCausalLM)
