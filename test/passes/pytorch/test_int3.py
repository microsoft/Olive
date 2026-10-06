# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file
from transformers import LlamaConfig, LlamaForCausalLM

from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.inc_quantization import IncQuantization
from olive.passes.pytorch.autoawq import AutoAWQQuantizer
from olive.passes.pytorch.autoclip import AutoClip
from olive.passes.pytorch.autogptq import GptqQuantizer
from olive.passes.pytorch.gptq import Gptq, gptq_quantize_weight
from olive.passes.pytorch.gptqmodel import GptqModel
from olive.passes.pytorch.kquant import KQuant
from olive.passes.pytorch.rtn import Rtn
from olive.passes.pytorch.selective_mixed_precision import SelectiveMixedPrecision
from test.passes.pytorch.test_rtn import _make_local_tiny_qwen3_moe, _save_trivial_tokenizer


@pytest.fixture(name="int3_input_model")
def int3_input_model_fixture(tmp_path: Path) -> HfModelHandler:
    torch.manual_seed(0)
    config = LlamaConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32,
    )
    path = tmp_path / "input"
    model = LlamaForCausalLM(config)
    model.save_pretrained(path)
    _save_trivial_tokenizer(path, config.vocab_size)
    return HfModelHandler(model_path=path, load_kwargs={"attn_implementation": "eager"})


@pytest.mark.parametrize("pass_class", [Rtn, KQuant, Gptq])
@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("group_size", [-1, 16])
def test_native_int3_pass_saves_and_reloads_mixed_widths(
    int3_input_model, tmp_path, monkeypatch, pass_class, sym, group_size
):
    generator = torch.Generator().manual_seed(1)
    data = [
        {"input_ids": torch.randint(0, 32, (1, 8), generator=generator), "attention_mask": torch.ones(1, 8)}
        for _ in range(2)
    ]
    monkeypatch.setattr("olive.passes.pytorch.quant_utils.get_calibration_dataset", lambda *_args, **_kwargs: data)
    config = {
        "bits": 3,
        "group_size": group_size,
        "sym": sym,
        "lm_head": True,
        "overrides": {
            "model.layers.0.self_attn.q_proj": {"bits": 8},
            "re:.*\\.self_attn\\.o_proj": {"bits": 4},
        },
    }
    if pass_class is not Gptq:
        config["embeds"] = True
    quant_pass = create_pass_from_dict(
        pass_class,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider"),
    )
    out = quant_pass.run(int3_input_model, tmp_path / "quantized")
    loaded = out.load_model()
    assert loaded.config.quantization_config.bits == 3
    assert loaded.model.layers[0].self_attn.q_proj.weight.bits == 8
    assert loaded.model.layers[0].self_attn.o_proj.weight.bits == 4
    quant = loaded.model.layers[0].mlp.down_proj.weight
    assert isinstance(quant, QuantTensor)
    assert quant.bits == 3
    assert quant.qweight.shape == (16, 12)
    assert quant.qzeros is None if sym else quant.qzeros is not None
    assert isinstance(loaded.model.embed_tokens.weight, QuantTensor) == (pass_class is not Gptq)

    saved = load_file(str(tmp_path / "quantized" / "model.safetensors"))
    for name, param in loaded.named_parameters():
        if not isinstance(param, QuantTensor):
            continue
        for suffix, tensor in [("_qweight", param.qweight), ("_scales", param.scales), ("_qzeros", param.qzeros)]:
            if tensor is not None:
                torch.testing.assert_close(saved[name + suffix], tensor, rtol=0, atol=0)
    config_json = json.loads((tmp_path / "quantized" / "config.json").read_text())
    assert config_json["quantization_config"]["bits"] == 3

    tokens = torch.tensor([[1, 2, 3]])
    with torch.no_grad():
        expected = loaded(tokens).logits
    loaded.save_pretrained(tmp_path / "resaved")
    reloaded = HfModelHandler(
        model_path=tmp_path / "resaved", load_kwargs={"attn_implementation": "eager"}
    ).load_model()
    with torch.no_grad():
        torch.testing.assert_close(reloaded(tokens).logits, expected, rtol=0, atol=0)
    for name, param in reloaded.named_parameters():
        if isinstance(param, QuantTensor):
            source = loaded.get_parameter(name)
            assert torch.equal(param.qweight, source.qweight)
            assert param.qweight is reloaded.get_submodule(name.rsplit(".", 1)[0]).get_buffer("weight_qweight")


@pytest.mark.parametrize("pass_class", [Rtn, KQuant])
def test_native_pass_preserves_int3_when_quantizing_remaining_weights(int3_input_model, tmp_path, pass_class):
    first = create_pass_from_dict(pass_class, {"bits": 3, "group_size": 16}, disable_search=True)
    quantized = first.run(int3_input_model, tmp_path / "first")
    before = quantized.load_model().model.layers[0].self_attn.q_proj.weight.qweight.clone()
    second = create_pass_from_dict(pass_class, {"bits": 8, "group_size": 16, "embeds": True}, disable_search=True)
    result = second.run(quantized, tmp_path / "second").load_model()
    assert result.model.layers[0].self_attn.q_proj.weight.bits == 3
    assert torch.equal(result.model.layers[0].self_attn.q_proj.weight.qweight, before)
    assert result.model.embed_tokens.weight.bits == 8


@pytest.mark.parametrize("pass_class", [Rtn, KQuant, Gptq])
@pytest.mark.parametrize("sym", [False, True])
def test_native_int3_moe_saves_reloads_and_runs(tmp_path, monkeypatch, pass_class, sym):
    model = _make_local_tiny_qwen3_moe(tmp_path / "input")
    tokens = torch.tensor([[1, 2, 3, 4]])
    monkeypatch.setattr(
        "olive.passes.pytorch.quant_utils.get_calibration_dataset",
        lambda *_args, **_kwargs: [{"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}],
    )
    quant_pass = create_pass_from_dict(
        pass_class, {"bits": 3, "moe": True, "group_size": -1, "sym": sym}, disable_search=True
    )
    out = quant_pass.run(model, tmp_path / "quantized")
    loaded = out.load_model()
    experts = loaded.model.layers[0].mlp.experts
    for name in ["gate_up_proj", "down_proj"]:
        param = experts.get_parameter(name)
        assert isinstance(param, QuantTensor)
        assert param.bits == 3
        assert param.dim() == 3
        assert param.qweight.shape[-1] == (param.shape[-1] * 3 + 7) // 8
    # The grouped_mm implementation transposes QuantTensor weights; select the supported eager path.
    loaded.config._experts_implementation = "eager"
    with torch.no_grad():
        assert torch.isfinite(loaded(tokens).logits).all()
    before = experts.gate_up_proj.qweight.clone()
    reloaded = out.load_model(cache_model=False)
    assert torch.equal(reloaded.model.layers[0].mlp.experts.gate_up_proj.qweight, before)


def test_gptq_rejects_already_int3_quantized_input(int3_input_model, tmp_path):
    quantized = create_pass_from_dict(Rtn, {"bits": 3, "group_size": 16}, disable_search=True).run(
        int3_input_model, tmp_path / "quantized"
    )
    gptq = create_pass_from_dict(Gptq, {"bits": 3, "group_size": 16}, disable_search=True)
    with pytest.raises(ValueError, match="quantized"):
        gptq.run(quantized, tmp_path / "gptq")


@pytest.mark.parametrize("algorithm", ["snr", "high_precision_lm_head"])
def test_selective_mixed_precision_propagates_int3_to_rtn(int3_input_model, tmp_path, algorithm):
    smp = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": algorithm, "bits": 3, "high_bits": 4, "group_size": 16, "ratio": 0.25},
        disable_search=True,
    )
    selected = smp.run(int3_input_model, tmp_path / "selection")
    assert selected.model_attributes["mixed_precision_info"]["default"]["bits"] == 3
    out = create_pass_from_dict(Rtn, {"group_size": 16, "lm_head": True}, disable_search=True).run(
        selected, tmp_path / "quantized"
    )
    widths = {param.bits for param in out.load_model().parameters() if isinstance(param, QuantTensor)}
    assert widths == {3, 4}


@pytest.mark.parametrize("sym", [False, True])
@pytest.mark.parametrize("group_size", [-1, 16])
def test_gptq_int3_output_matches_materialized_quant_tensor(sym, group_size):
    generator = torch.Generator().manual_seed(2)
    weight = torch.randn(4, 32, generator=generator)
    activations = torch.randn(32, 64, generator=generator)
    hessian = activations @ activations.t()
    quantizer = WeightQuantizer(bits=3, symmetric=sym, group_size=group_size)
    quantized, scales, zeros = gptq_quantize_weight(weight, hessian, quantizer)
    qt = QuantTensor.from_float(
        quantized, bits=3, symmetric=sym, group_size=group_size, scales=scales, zero_points=zeros
    )
    torch.testing.assert_close(qt.to_dense(), quantized)


@pytest.mark.parametrize("pass_class", [GptqQuantizer, GptqModel, AutoAWQQuantizer])
def test_external_quantizer_rejects_native_int3_before_importing_backend(pass_class):
    quant_pass = create_pass_from_dict(pass_class, {"bits": 3}, disable_search=True)
    with pytest.raises(ValueError, match="INT3"):
        quant_pass._run_for_config(None, quant_pass.config, "unused")


def test_inc_weight_only_config_rejects_native_int3():
    quant_pass = create_pass_from_dict(
        IncQuantization,
        {"approach": "weight_only", "weight_only_config": {"bits": 3}, "data_config": {"name": "unused"}},
        disable_search=True,
    )
    with pytest.raises(ValueError, match="INT3 is supported only"):
        quant_pass._set_woq_config(quant_pass.config.model_dump())


def test_gptqmodel_rejects_dynamic_int3_before_importing_backend():
    quant_pass = create_pass_from_dict(
        GptqModel, {"bits": 4, "dynamic": {"re:.*proj": {"bits": 3}}}, disable_search=True
    )
    with pytest.raises(ValueError, match="INT3"):
        quant_pass._run_for_config(None, quant_pass.config, "unused")


def test_autoclip_int3_settings_can_precede_rtn(int3_input_model, tmp_path, monkeypatch):
    tokens = torch.tensor([[1, 2, 3, 4]])
    monkeypatch.setattr(
        "olive.passes.pytorch.quant_utils.get_calibration_dataset",
        lambda *_args, **_kwargs: [{"input_ids": tokens, "attention_mask": torch.ones_like(tokens)}],
    )
    clipping = create_pass_from_dict(
        AutoClip, {"bits": 3, "group_size": 16, "n_grid": 4, "n_sample_token": 4}, disable_search=True
    )
    assert AutoClip.validate_config(clipping.config, clipping.accelerator_spec)
    clipped = clipping.run(int3_input_model, tmp_path / "clipped")
    assert not any(isinstance(param, QuantTensor) for param in clipped.load_model().parameters())
    quantized = create_pass_from_dict(Rtn, {"bits": 3, "group_size": 16}, disable_search=True).run(
        clipped, tmp_path / "quantized"
    )
    assert quantized.load_model().model.layers[0].mlp.down_proj.weight.bits == 3
