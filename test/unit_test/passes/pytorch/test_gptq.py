# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model.handler.pytorch import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import GptqQuantizer


def get_dummy_dataloader_func():
    return [
        {
            "input_ids": torch.randint(10, 100, (1, 128), dtype=torch.long),
            "attention_mask": torch.ones(1, 128, dtype=torch.long),
        }
        for _ in range(128)
    ]


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gptq requires GPU.",
)
def test_gptq_default(tmp_path: Path):
    # setup
    input_model = PyTorchModelHandler(
        hf_config={
            "model_class": "OPTForCausalLM",
            "model_name": "facebook/opt-125m",
            "task": "text-generation",
        }
    )
    config = {"dataloader_func": get_dummy_dataloader_func}

    p = create_pass_from_dict(
        GptqQuantizer,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)
    assert out is not None
