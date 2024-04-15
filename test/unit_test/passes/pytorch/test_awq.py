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
from olive.passes.pytorch.awq import AwqQuantizer


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="gptq requires GPU.",
)
@pytest.mark.parametrize("pack_model_for_onnx_conversion", [True, False])
def test_awq(pack_model_for_onnx_conversion, tmp_path: Path):
    # setup
    input_model = PyTorchModelHandler(
        hf_config={
            "model_class": "OPTForCausalLM",
            "model_name": "facebook/opt-125m",
            "task": "text-generation",
            "from_pretrained_args": {"extra_args": {"use_safetensors": False}},
        }
    )

    p = create_pass_from_dict(
        AwqQuantizer,
        {"pack_model_for_onnx_conversion": pack_model_for_onnx_conversion},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, None, gptq_out_folder)
    assert out is not None
