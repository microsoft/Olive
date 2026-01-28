# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.rtn import Rtn
from test.utils import get_tiny_phi3


@pytest.mark.parametrize("group_size", [-1, 16])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("lm_head", [True, False])
def test_gptq(tmp_path: Path, group_size: int, sym: bool, lm_head: bool):
    # setup
    input_model = get_tiny_phi3()
    p = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": group_size,
            "lm_head": lm_head,
            "sym": sym,
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
    assert loaded_model.__class__.__name__ == "Phi3ForCausalLM"
    assert hasattr(loaded_model, "quantization_method")
    assert loaded_model.quantization_method == "olive"
    assert hasattr(loaded_model.config, "quantization_config")
    assert isinstance(loaded_model.config.quantization_config, OliveHfQuantizationConfig)
    assert loaded_model.config.quantization_config.group_size == group_size
    assert not any(isinstance(m, torch.nn.Linear) for m in loaded_model.model.layers.modules())
    assert isinstance(loaded_model.model.layers[0].self_attn.o_proj, QuantLinear)
    assert loaded_model.model.layers[0].self_attn.o_proj.quantizer.bits == 8
    assert loaded_model.model.layers[0].mlp.down_proj.quantizer.bits == 4
    assert loaded_model.config.quantization_config.lm_head == lm_head
    assert isinstance(loaded_model.lm_head, QuantLinear) == lm_head
    assert isinstance(loaded_model.model.embed_tokens, torch.nn.Embedding)

    # compose another rtn pass on top of the partially quantized model
    p2 = create_pass_from_dict(
        Rtn,
        {
            "bits": 8,
            "group_size": group_size,
            "lm_head": True,
            "embeds": True,
            "sym": sym,
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder_2 = str(tmp_path / "gptq2")
    out2 = p2.run(out, gptq_out_folder_2)

    # assert
    assert isinstance(out2, HfModelHandler)
    loaded_model_2 = out2.load_model()
    # check that the embed tokens layer is quantized to 8 bits
    assert isinstance(loaded_model_2.model.embed_tokens, QuantEmbedding)
    assert loaded_model_2.model.embed_tokens.quantizer.bits == 8
    # check that the lm head is quantized to 8 bits if it was not quantized before
    assert isinstance(loaded_model_2.lm_head, QuantLinear)
    assert loaded_model_2.lm_head.quantizer.bits == 4 if lm_head else 8
