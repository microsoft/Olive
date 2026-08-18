# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer, get_maxq_minq
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.kquant import KQuant, kquant_find_qparams
from olive.passes.pytorch.moe_support import MoeSupportError
from olive.passes.pytorch.quant_utils import prepare_model
from test.utils import get_tiny_phi3


def _save_trivial_tokenizer(save_path: Path, vocab_size: int) -> None:
    """Save a local tokenizer so pass metadata serialization never needs the hub."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokenizer = Tokenizer(models.WordLevel({f"t{i}": i for i in range(vocab_size)}, unk_token="t0"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=tokenizer, unk_token="t0", pad_token="t0").save_pretrained(save_path)


def _make_local_tiny_qwen3_moe(save_path: Path) -> HfModelHandler:
    """Save a tiny K-last fused-experts model without downloading a checkpoint."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    save_path.mkdir(parents=True, exist_ok=True)
    config = Qwen3MoeConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=16,
        intermediate_size=16,
        moe_intermediate_size=8,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        num_experts=2,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        head_dim=8,
        experts_implementation="eager",
    )
    Qwen3MoeForCausalLM(config).save_pretrained(save_path)
    _save_trivial_tokenizer(save_path, config.vocab_size)
    return HfModelHandler(model_path=str(save_path))


def _is_quant(module: torch.nn.Module) -> bool:
    if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        return False
    weight = module._parameters.get("weight")
    return weight is not None and isinstance(weight.data, QuantTensor)


def _bits(module: torch.nn.Module) -> int:
    return module.weight.data.bits


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


@pytest.mark.parametrize("sym", [True, False])
def test_kquant_find_qparams_3d_matches_per_expert_results(sym: bool):
    torch.manual_seed(1)
    weight = torch.randn(3, 5, 32, dtype=torch.float32)
    group_size = 8
    maxq, minq = get_maxq_minq(4, signed=False)

    scales, zero_points = kquant_find_qparams(weight, group_size, maxq, minq, symmetric=sym)

    assert scales.shape == (3, 5, 4)
    assert zero_points.shape == (3, 5, 4)
    for expert_idx in range(weight.shape[0]):
        expert_scales, expert_zero_points = kquant_find_qparams(
            weight[expert_idx], group_size, maxq, minq, symmetric=sym
        )
        torch.testing.assert_close(scales[expert_idx], expert_scales)
        torch.testing.assert_close(zero_points[expert_idx], expert_zero_points)


@pytest.mark.parametrize(
    ("layout", "message"),
    [
        (True, r"\(E, K, OUT\)"),
        (None, "is_transposed"),
        ("False", "is_transposed"),
        ("missing", "is_transposed"),
    ],
)
def test_kquant_moe_rejects_unsafe_or_unverifiable_layout(tmp_path: Path, monkeypatch, layout: object, message: str):
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")

    def patched_prepare_model(*args, **kwargs):
        wrapper, qcfg, retie = prepare_model(*args, **kwargs)
        for layer in wrapper.get_layer_wrappers():
            experts = layer.get_experts(return_name=False)
            if experts is None:
                continue
            if layout == "missing":
                if hasattr(experts, "is_transposed"):
                    del experts.is_transposed
            else:
                experts.is_transposed = layout
        return wrapper, qcfg, retie

    monkeypatch.setattr("olive.passes.pytorch.kquant.prepare_model", patched_prepare_model)
    quantizer = create_pass_from_dict(KQuant, {"moe": True, "group_size": -1}, disable_search=True)

    with pytest.raises(MoeSupportError, match=message):
        quantizer.run(input_model, str(tmp_path / "kquant"))


def test_kquant_moe_module_list_without_direct_3d_parameter_is_exempt(tmp_path: Path, monkeypatch):
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")

    def patched_prepare_model(*args, **kwargs):
        wrapper, qcfg, retie = prepare_model(*args, **kwargs)
        classic_experts = torch.nn.ModuleList([torch.nn.Linear(4, 4) for _ in range(2)])
        for layer in wrapper.get_layer_wrappers():
            layer.get_experts = lambda return_name=True, experts=classic_experts: (
                (experts, "mlp.experts") if return_name else experts
            )
        return wrapper, qcfg, retie

    monkeypatch.setattr("olive.passes.pytorch.kquant.prepare_model", patched_prepare_model)
    quantizer = create_pass_from_dict(KQuant, {"moe": True, "group_size": -1}, disable_search=True)

    out = quantizer.run(input_model, str(tmp_path / "kquant"))

    assert isinstance(out, HfModelHandler)


def test_kquant_moe_k_last_quantize_dequantize_roundtrip(tmp_path: Path):
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")
    original = input_model.load_model().model.layers[0].mlp.experts.gate_up_proj.detach().clone()
    quantizer = create_pass_from_dict(
        KQuant,
        {"bits": 4, "group_size": 4, "moe": True},
        disable_search=True,
    )

    out = quantizer.run(input_model, str(tmp_path / "kquant"))

    experts = out.load_model().model.layers[0].mlp.experts
    quantized = experts.gate_up_proj.data
    assert isinstance(quantized, QuantTensor)
    assert quantized.scales.shape == (*original.shape[:-1], original.shape[-1] // 4)
    dequantized = quantized.to_dense()
    assert torch.isfinite(dequantized).all()
    error = (dequantized - original).abs().mean()
    relative_error = error / original.abs().mean()
    assert 0 < relative_error < 0.15


def test_kquant_moe_false_does_not_run_layout_gate(tmp_path: Path, monkeypatch):
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")

    def unexpected_gate(*args, **kwargs):
        pytest.fail("MoE layout support check must not run when moe=False")

    monkeypatch.setattr("olive.passes.pytorch.kquant.check_moe_layout_support", unexpected_gate)
    quantizer = create_pass_from_dict(KQuant, {"moe": False, "group_size": -1}, disable_search=True)

    out = quantizer.run(input_model, str(tmp_path / "kquant"))

    experts = out.load_model().model.layers[0].mlp.experts
    assert not any(isinstance(param.data, QuantTensor) for param in experts.parameters())


def test_kquant_moe_gate_ignores_prior_checkpoint_moe_flag(tmp_path: Path, monkeypatch):
    """Regression test: a second KQuant pass with moe=False must not re-run the layout gate.

    ``prepare_model`` ORs a pre-existing checkpoint's ``moe`` flag into the merged
    ``qcfg.moe`` (see ``quant_utils.prepare_model``), so gating on ``qcfg.moe`` would make
    this second, moe=False invocation incorrectly re-run fused-experts layout validation
    -- something this run never asked for. The gate must key off this invocation's own
    ``config.moe`` request instead.
    """
    input_model = _make_local_tiny_qwen3_moe(tmp_path / "input_model")
    first_pass = create_pass_from_dict(KQuant, {"bits": 4, "group_size": 4, "moe": True}, disable_search=True)
    quantized = first_pass.run(input_model, str(tmp_path / "kquant_first"))

    def unexpected_gate(*args, **kwargs):
        pytest.fail("MoE layout support check must not re-run when this invocation requests moe=False")

    monkeypatch.setattr("olive.passes.pytorch.kquant.check_moe_layout_support", unexpected_gate)
    second_pass = create_pass_from_dict(KQuant, {"moe": False, "lm_head": True, "group_size": -1}, disable_search=True)
    out = second_pass.run(quantized, str(tmp_path / "kquant_second"))

    loaded = out.load_model()
    assert isinstance(loaded.lm_head.weight.data, QuantTensor)


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
    assert not any(isinstance(m, torch.nn.Linear) and not _is_quant(m) for m in loaded_model.model.layers.modules())
    assert _is_quant(loaded_model.model.layers[0].self_attn.o_proj)
    assert _bits(loaded_model.model.layers[0].self_attn.o_proj) == 8
    assert _bits(loaded_model.model.layers[0].mlp.down_proj) == 4
    assert _is_quant(loaded_model.lm_head) == lm_head
    assert isinstance(loaded_model.model.embed_tokens, torch.nn.Embedding)
    assert not _is_quant(loaded_model.model.embed_tokens)

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
    assert _is_quant(loaded_model_2.model.embed_tokens)
    assert _bits(loaded_model_2.model.embed_tokens) == 8
    assert _is_quant(loaded_model_2.lm_head)
    assert _bits(loaded_model_2.lm_head) == (4 if lm_head else 8)
