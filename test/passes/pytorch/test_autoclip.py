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
from olive.passes.pytorch.autoclip import AutoClip
from test.utils import get_tiny_phi3, make_local_tiny_llama


# running on CPU takes time so will only run a subset of tests when GPU is not available
@pytest.mark.parametrize(
    ("model_path", "expected_model_type"),
    [
        ("tiny-phi3", "Phi3ForCausalLM"),
        ("tiny-llama", "LlamaForCausalLM"),
    ],
)
@pytest.mark.parametrize("group_size", [-1, 16] if torch.cuda.is_available() else [16])
def test_autoclip(tmp_path: Path, model_path: str, expected_model_type: str, group_size: int):
    # setup
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "input_model")
    else:
        input_model = get_tiny_phi3()
    p = create_pass_from_dict(
        AutoClip,
        {
            "group_size": group_size,
            "lm_head": False,
            "sym": False,
            "overrides": {"model.layers.0.self_attn.o_proj": {"bits": 8}},
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)
    loaded_model = out.load_model()
    assert loaded_model.__class__.__name__ == expected_model_type
