# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.common.quant.utils import WeightQuantizer, get_maxq_minq
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.kquant import KQuant, kquant_find_qparams
from test.utils import get_tiny_phi3


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("bits", [2, 4])
def test_kquant_find_qparams_beats_min_max_rtn(bits: int, sym: bool):
    torch.manual_seed(0)
    weight = torch.randn(8, 64, dtype=torch.float32)
    group_size = 32
    maxq, minq = get_maxq_minq(bits, signed=False)

    kq_scales, kq_zp = kquant_find_qparams(weight, group_size=group_size, maxq=maxq, minq=minq, symmetric=sym)
    quantizer = WeightQuantizer(bits=bits, symmetric=sym, group_size=group_size)
    err_kq = (quantizer.fake_quantize(weight, kq_scales, kq_zp) - weight).abs().mean().item()

    rtn_scales, rtn_zp = quantizer.find_qparams(weight)
    err_rtn = (quantizer.fake_quantize(weight, rtn_scales, rtn_zp) - weight).abs().mean().item()

    assert err_kq < err_rtn


@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("values", [0.0, 100.0, -7.5])
def test_kquant_find_qparams_handles_constant_groups(values: float, sym: bool):
    weight = torch.full((4, 32), values, dtype=torch.float32)
    bits = 4
    group_size = 16
    maxq, minq = get_maxq_minq(bits, signed=False)

    scales, zero_points = kquant_find_qparams(weight, group_size=group_size, maxq=maxq, minq=minq, symmetric=sym)
    quantizer = WeightQuantizer(bits=bits, symmetric=sym, group_size=group_size)
    dq = quantizer.fake_quantize(weight, scales, zero_points)
    assert torch.isfinite(dq).all()
    assert torch.allclose(dq, weight, atol=1e-5)


@pytest.mark.parametrize("group_size", [-1, 16])
@pytest.mark.parametrize("sym", [True, False])
@pytest.mark.parametrize("lm_head", [True, False])
def test_kquant(tmp_path: Path, group_size: int, sym: bool, lm_head: bool):
    input_model = get_tiny_phi3()
    p = create_pass_from_dict(
        KQuant,
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
    out_folder = str(tmp_path / "kquant")

    out = p.run(input_model, out_folder)

    assert isinstance(out, HfModelHandler)
    loaded_model = out.load_model()
    assert loaded_model.__class__.__name__ == "Phi3ForCausalLM"
    assert hasattr(loaded_model, "quantization_method")
    assert loaded_model.quantization_method == "olive"
    assert isinstance(loaded_model.config.quantization_config, OliveHfQuantizationConfig)
    assert loaded_model.config.quantization_config.symmetric is sym
    assert loaded_model.config.quantization_config.group_size == group_size
    assert loaded_model.config.quantization_config.lm_head == lm_head
    assert not any(isinstance(m, torch.nn.Linear) for m in loaded_model.model.layers.modules())
    assert isinstance(loaded_model.model.layers[0].self_attn.o_proj, QuantLinear)
    assert loaded_model.model.layers[0].self_attn.o_proj.quantizer.bits == 8
    assert loaded_model.model.layers[0].mlp.down_proj.quantizer.bits == 4
    assert isinstance(loaded_model.lm_head, QuantLinear) == lm_head
    assert isinstance(loaded_model.model.embed_tokens, torch.nn.Embedding)

    # compose another kquant pass to also quantize embeds and lm_head
    p2 = create_pass_from_dict(
        KQuant,
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
    out2 = p2.run(out, str(tmp_path / "kquant2"))

    assert isinstance(out2, HfModelHandler)
    loaded_model_2 = out2.load_model()
    assert isinstance(loaded_model_2.model.embed_tokens, QuantEmbedding)
    assert loaded_model_2.model.embed_tokens.quantizer.bits == 8
    assert isinstance(loaded_model_2.lm_head, QuantLinear)
    assert loaded_model_2.lm_head.quantizer.bits == 4 if lm_head else 8
