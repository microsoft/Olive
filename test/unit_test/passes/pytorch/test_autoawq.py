# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler, PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.autoawq import AutoAWQQuantizer


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="awq requires GPU.",
)
@pytest.mark.parametrize("pack_model_for_onnx_conversion", [True, False])
def test_awq(pack_model_for_onnx_conversion, tmp_path: Path):
    # setup
    input_model = HfModelHandler(model_path="facebook/opt-125m", load_kwargs={"use_safetensors": False})

    p = create_pass_from_dict(
        AutoAWQQuantizer,
        {"pack_model_for_onnx_conversion": pack_model_for_onnx_conversion},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)

    # assert
    if pack_model_for_onnx_conversion:
        assert isinstance(out, PyTorchModelHandler)
    else:
        assert isinstance(out, HfModelHandler)

    from transformers import OPTForCausalLM

    assert isinstance(out.load_model(), OPTForCausalLM)
