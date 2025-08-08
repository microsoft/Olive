# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.linear import QuantLinear
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import Gptq
from test.utils import make_local_tiny_llama


@pytest.mark.parametrize(
    ("model_path", "expected_model_type"),
    [
        ("katuni4ka/tiny-random-phi3", "Phi3ForCausalLM"),
        ("tiny-llama", "LlamaForCausalLM"),
    ],
)
@pytest.mark.parametrize("group_size", [-1, 16])
def test_gptq(tmp_path: Path, model_path: str, expected_model_type: str, group_size: int):
    # setup
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "input_model")
    else:
        input_model = HfModelHandler(model_path=model_path)
    p = create_pass_from_dict(
        Gptq,
        {"group_size": group_size},
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
    assert hasattr(loaded_model, "quantization_method")
    assert loaded_model.quantization_method == "olive"
    assert hasattr(loaded_model.config, "quantization_config")
    assert isinstance(loaded_model.config.quantization_config, OliveHfQuantizationConfig)
    assert loaded_model.config.quantization_config.group_size == group_size
    assert not any(isinstance(m, torch.nn.Linear) for m in loaded_model.model.layers.modules())
    assert isinstance(loaded_model.model.layers[0].self_attn.o_proj, QuantLinear)
