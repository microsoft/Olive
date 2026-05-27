# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import importlib.util
import math
import sys
from copy import deepcopy
from types import ModuleType, SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.utils import WeightQuantizer
from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import selective_mixed_precision as smp_module
from olive.passes.pytorch.quant_utils import get_qkv_quantization_groups
from olive.passes.pytorch.selective_mixed_precision import (
    KldMemoryMode,
    SelectiveMixedPrecision,
    _IqeRelativeStrategy,
    _IqeStrategy,
    _KldGradientStrategy,
    _SnrRelativeStrategy,
    _SnrStrategy,
)
from olive.passes.pytorch.train_utils import kl_div_loss
from test.utils import get_tiny_phi3

# ---------------------------------------------------------------------------
# Test fixtures / helpers
# ---------------------------------------------------------------------------


class KldGradientTestModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(4, 4, bias=False)
        self.out = torch.nn.Linear(4, 4, bias=False)
        self.gradient_checkpointing_enabled = False

        with torch.no_grad():
            self.proj.weight.copy_(
                torch.tensor(
                    [
                        [0.10, -0.20, 0.30, -0.40],
                        [0.50, 0.60, -0.70, -0.80],
                        [-0.15, 0.25, -0.35, 0.45],
                        [0.55, -0.65, 0.75, -0.85],
                    ]
                )
            )
            self.out.weight.copy_(
                torch.tensor(
                    [
                        [0.35, -0.45, 0.55, -0.65],
                        [-0.75, 0.85, -0.95, 1.05],
                        [0.12, 0.22, -0.32, -0.42],
                        [-0.52, 0.62, 0.72, -0.82],
                    ]
                )
            )

    def gradient_checkpointing_enable(self):
        self.gradient_checkpointing_enabled = True

    def forward(self, tokens):
        hidden_states = torch.tanh(self.proj(tokens))
        return SimpleNamespace(logits=self.out(hidden_states))


def _wrap(model: torch.nn.Module, layer_prefix: str = "model.layers"):
    """Lightweight stand-in for ``ModelWrapper`` exposing ``.model`` and ``.get_layers``."""
    return SimpleNamespace(
        model=model,
        get_layers=lambda return_name=True: (None, layer_prefix) if return_name else None,
    )


def get_kld_gradient_test_data():
    return [
        {
            "tokens": torch.tensor(
                [
                    [[0.20, -0.70, 1.10, 0.50], [0.30, 0.40, -0.90, 0.80]],
                    [[-0.60, 0.10, 0.70, -0.20], [0.90, -0.30, 0.20, -0.50]],
                ]
            )
        },
        {
            "tokens": torch.tensor(
                [
                    [[-0.40, 0.60, -0.10, 0.30], [0.50, -0.80, 0.40, 0.20]],
                    [[0.70, 0.20, -0.60, 0.10], [-0.30, 0.90, -0.50, -0.70]],
                ]
            )
        },
    ]


def get_kld_gradient_quantizers():
    return WeightQuantizer(bits=4, group_size=4, symmetric=False), WeightQuantizer(bits=8, group_size=4, symmetric=True)


def get_legacy_kld_scores(model, data, quantizer, high_quantizer, device):
    """Produce ``(unit_numels, unit_scores)`` with singleton units (reference implementation)."""
    model.to(device).eval()
    quantized_model = deepcopy(model).to(device).eval()

    for parameter in quantized_model.parameters():
        parameter.requires_grad = False
    quantized_model.gradient_checkpointing_enable()

    module_numels = {}
    quantized_layers = {}
    grad_accum = {}

    with torch.no_grad():
        for module_name, module in quantized_model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue

            module_numels[module_name] = module.weight.numel()
            module.weight.data = quantizer.fake_quantize(module.weight.data)
            module.weight.requires_grad = True
            quantized_layers[module_name] = module
            grad_accum[module_name] = torch.zeros_like(module.weight.data, dtype=torch.float32)

    for batch in data:
        inputs = {key: value.to(device) for key, value in batch.items()}

        with torch.no_grad():
            teacher_logits = model(**inputs).logits

        student_logits = quantized_model(**inputs).logits
        loss = kl_div_loss(student_logits, teacher_logits).mean()
        loss.backward()

        for module_name, layer in quantized_layers.items():
            grad_accum[module_name] += layer.weight.grad.data.detach().float()
        quantized_model.zero_grad()

    module_stats = {}
    with torch.no_grad():
        for module_name, layer in quantized_layers.items():
            grad = grad_accum[module_name] / len(data)
            high_weight = high_quantizer.fake_quantize(model.get_submodule(module_name).weight.data)
            alignment = (grad * (layer.weight.data - high_weight)).sum().item()
            module_stats[module_name] = {"alignment": alignment}

    # Reuse the strategy's aggregation so the reference matches schema-wise.
    return _KldGradientStrategy(quantizer, high_quantizer)._aggregate(module_numels, module_stats, qkv_groups=())


def patch_kld_calibration_data(monkeypatch, data):
    monkeypatch.setattr(
        "olive.passes.pytorch.selective_mixed_precision.get_calibration_dataset",
        lambda *_args, **_kwargs: data,
    )


def assert_scores_close(actual_scores, expected_scores):
    """Compare per-unit scalar sensitivity scores."""
    assert actual_scores.keys() == expected_scores.keys()
    for unit, expected in expected_scores.items():
        assert actual_scores[unit] == pytest.approx(expected, rel=1e-6, abs=1e-6)


@pytest.fixture(name="input_model", scope="module")
def input_model_fixture(tmp_path_factory):
    save_path = tmp_path_factory.mktemp("selective-mixed-precision-test")
    model = LlamaForCausalLM(
        LlamaConfig(  # pylint: disable=unexpected-keyword-arg
            hidden_size=16,
            intermediate_size=64,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=4,
            vocab_size=32000,
        )
    )
    model.save_pretrained(save_path)
    return HfModelHandler(save_path)


# ---------------------------------------------------------------------------
# End-to-end pass behavior
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("algorithm", "expected_layer_indices", "include_qkv"),
    [
        ("k_quant_down", [0, 3, 6, 7], False),  # first 1/8, every 3rd, and last 1/8
        ("k_quant_mixed", [0, 3, 6, 7], True),
        ("k_quant_last", [], False),
    ],
)
def test_selective_mixed_precision_k_quant(algorithm, expected_layer_indices, include_qkv, input_model, tmp_path):
    """End-to-end: rule-based k_quant_* algorithms write the expected mixed_precision_info."""
    config = {"algorithm": algorithm}
    p = create_pass_from_dict(SelectiveMixedPrecision, config, disable_search=True)

    output_model = p.run(input_model, str(tmp_path))

    assert "mixed_precision_info" in output_model.model_attributes
    expected_mp_info = {
        "default": {"bits": PrecisionBits.BITS4},
        "overrides": {
            "lm_head": {"bits": 8},
        },
    }
    for idx in expected_layer_indices:
        expected_mp_info["overrides"].update(
            {
                **(
                    {
                        f"model.layers.{idx}.self_attn.q_proj": {"bits": PrecisionBits.BITS8},
                        f"model.layers.{idx}.self_attn.k_proj": {"bits": PrecisionBits.BITS8},
                        f"model.layers.{idx}.self_attn.v_proj": {"bits": PrecisionBits.BITS8},
                    }
                    if include_qkv
                    else {}
                ),
                f"model.layers.{idx}.mlp.down_proj": {"bits": PrecisionBits.BITS8},
            }
        )
    assert output_model.model_attributes["mixed_precision_info"] == expected_mp_info


@pytest.mark.parametrize("algorithm", ["snr", "snr_relative", "iqe", "iqe_relative", "kld_gradient"])
def test_selective_mixed_precision_scored(algorithm, tmp_path):
    """End-to-end: every score-based algorithm produces a valid ``mixed_precision_info``."""
    if algorithm == "kld_gradient" and not torch.cuda.is_available():
        pytest.skip("Skipping kld_gradient test as it runs slow on CPU.")

    p = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": algorithm, "ratio": 0.8, "group_size": 16, "high_sym": True},
        disable_search=True,
    )

    output_model = p.run(get_tiny_phi3(), str(tmp_path))

    assert "mixed_precision_info" in output_model.model_attributes
    assert output_model.model_attributes["mixed_precision_info"]["default"] == {
        "bits": PrecisionBits.BITS4,
        "group_size": 16,
        "symmetric": False,
    }
    # all layers are so small, lm_head is always included to reach ratio
    assert output_model.model_attributes["mixed_precision_info"]["overrides"]["lm_head"] == {
        "bits": PrecisionBits.BITS8,
        "group_size": 16,
        "symmetric": True,
    }


def test_selective_mixed_precision_scored_run_co_promotes_qkv(tmp_path):
    """End-to-end: when SMP promotes any q/k/v to high precision, all three must be promoted.

    QKV grouping is part of the pass contract (ONNX export fuses q/k/v into one matmul). This
    test exercises the full pass on tiny-phi3 and asserts the three projections are never split.
    """
    p = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": "iqe", "ratio": 0.999, "group_size": 16, "high_sym": True},
        disable_search=True,
    )

    output_model = p.run(get_tiny_phi3(), str(tmp_path))

    overrides = output_model.model_attributes["mixed_precision_info"]["overrides"]
    for layer_idx in range(2):
        prefix = f"model.layers.{layer_idx}.self_attn.qkv_proj"
        qkv_in_overrides = [
            f"{prefix}.q_proj" in overrides,
            f"{prefix}.k_proj" in overrides,
            f"{prefix}.v_proj" in overrides,
        ]
        assert qkv_in_overrides[0] == qkv_in_overrides[1] == qkv_in_overrides[2], (
            f"qkv split at layer {layer_idx}: {qkv_in_overrides}"
        )


def test_selective_mixed_precision_scored_rejects_per_tensor_group_size_when_qkv_grouped(input_model, tmp_path):
    """Per-tensor (group_size=0) is rejected when QKV groups exist because per-member aggregation isn't exact."""
    p = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": "iqe", "ratio": 0.5, "group_size": 0, "high_group_size": 16},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="per-tensor"):
        p.run(input_model, str(tmp_path))


def test_selective_mixed_precision_scored_rejects_per_tensor_high_group_size_when_qkv_grouped(input_model, tmp_path):
    """Per-tensor high precision (high_group_size=0) is also rejected for the same reason."""
    p = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": "iqe", "ratio": 0.5, "group_size": 16, "high_group_size": 0},
        disable_search=True,
    )

    with pytest.raises(ValueError, match="per-tensor"):
        p.run(input_model, str(tmp_path))


# ---------------------------------------------------------------------------
# get_overrides_from_scores (algorithm-agnostic)
# ---------------------------------------------------------------------------


def test_get_overrides_from_scores_promotes_lowest_scored_units_until_ratio_met():
    """Units are sorted by ascending score and promoted until threshold is reached."""
    unit_numels = {("a",): 10, ("b",): 10, ("c",): 10}
    unit_scores = {("a",): 1.0, ("b",): -5.0, ("c",): 0.0}

    overrides, high = SelectiveMixedPrecision.get_overrides_from_scores(
        unit_numels, unit_scores, high_override_config={"bits": PrecisionBits.BITS8}, ratio=0.7
    )

    # threshold = 30*0.3 = 9; promoting b (10) satisfies it.
    assert overrides == {"b": {"bits": PrecisionBits.BITS8}}
    assert high == 10


def test_get_overrides_from_scores_expands_qkv_unit_to_every_member():
    """A QKV-grouped unit is promoted atomically: all member names get the override."""
    unit_numels = {("q", "k", "v"): 3, ("down",): 100}
    unit_scores = {("q", "k", "v"): -10.0, ("down",): 5.0}

    overrides, high = SelectiveMixedPrecision.get_overrides_from_scores(
        unit_numels, unit_scores, high_override_config={"bits": PrecisionBits.BITS8}, ratio=0.99
    )

    # threshold = 103*0.01 ≈ 1.03; first unit (QKV, numel 3) satisfies it.
    assert overrides == {
        "q": {"bits": PrecisionBits.BITS8},
        "k": {"bits": PrecisionBits.BITS8},
        "v": {"bits": PrecisionBits.BITS8},
    }
    assert high == 3


# ---------------------------------------------------------------------------
# QKV grouping in scored-config plumbing
# ---------------------------------------------------------------------------


def test_scored_config_keeps_qkv_same_precision(input_model):
    """Score-based selection must promote q/k/v as a single group even if only one is sensitive.

    The strategy aggregates per-projection IQE stats so the trio always shares precision.
    """
    model_wrapper = ModelWrapper.from_model(input_model.load_model())
    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"
    down_proj = "model.layers.0.mlp.down_proj"

    module_numels = {q_proj: 1, k_proj: 1, v_proj: 1, down_proj: 100}
    # k_proj alone has the largest iqe_raw (most sensitive); q/v are tiny but the group is
    # max-aggregated so the whole QKV unit ends up most sensitive overall.
    module_stats = {
        q_proj: {"iqe_raw": 0.1},
        k_proj: {"iqe_raw": 1.0},
        v_proj: {"iqe_raw": 0.1},
        down_proj: {"iqe_raw": 0.2},
    }

    qkv_groups = get_qkv_quantization_groups(model_wrapper, set(module_stats))
    strategy = _IqeStrategy(
        WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False),
        WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True),
    )
    unit_numels, unit_scores = strategy._aggregate(module_numels, module_stats, qkv_groups)

    overrides, high = SelectiveMixedPrecision.get_overrides_from_scores(
        unit_numels, unit_scores, high_override_config={"bits": PrecisionBits.BITS8}, ratio=0.98
    )

    assert overrides == {
        q_proj: {"bits": PrecisionBits.BITS8},
        k_proj: {"bits": PrecisionBits.BITS8},
        v_proj: {"bits": PrecisionBits.BITS8},
    }
    assert high == 3


def test_scored_config_ignores_single_packed_qkv():
    """A packed single qkv_proj (phi3 before unpacking) is not treated as a QKV group."""
    model_wrapper = ModelWrapper.from_model(get_tiny_phi3().load_model())

    assert not get_qkv_quantization_groups(model_wrapper, {"model.layers.0.self_attn.qkv_proj"})


def test_scored_config_groups_unpacked_qkv():
    """After ``maybe_unpack_qkv()``, the unpacked q/k/v submodules form a single group."""
    model_wrapper = ModelWrapper.from_model(get_tiny_phi3().load_model())
    model_wrapper.maybe_unpack_qkv()
    q_proj = "model.layers.0.self_attn.qkv_proj.q_proj"
    k_proj = "model.layers.0.self_attn.qkv_proj.k_proj"
    v_proj = "model.layers.0.self_attn.qkv_proj.v_proj"
    down_proj = "model.layers.0.mlp.down_proj"

    module_numels = {q_proj: 1, k_proj: 1, v_proj: 1, down_proj: 100}
    module_stats = {
        q_proj: {"iqe_raw": 0.1},
        k_proj: {"iqe_raw": 1.0},
        v_proj: {"iqe_raw": 0.1},
        down_proj: {"iqe_raw": 0.2},
    }

    qkv_groups = get_qkv_quantization_groups(model_wrapper, set(module_stats))
    strategy = _IqeStrategy(
        WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False),
        WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True),
    )
    unit_numels, unit_scores = strategy._aggregate(module_numels, module_stats, qkv_groups)

    overrides, high = SelectiveMixedPrecision.get_overrides_from_scores(
        unit_numels, unit_scores, high_override_config={"bits": PrecisionBits.BITS8}, ratio=0.98
    )

    assert overrides == {
        q_proj: {"bits": PrecisionBits.BITS8},
        k_proj: {"bits": PrecisionBits.BITS8},
        v_proj: {"bits": PrecisionBits.BITS8},
    }
    assert high == 3


# ---------------------------------------------------------------------------
# Per-strategy unit tests (combine_stats + score)
# ---------------------------------------------------------------------------


def _snr_strategy():
    return _SnrStrategy(quantizer=None, high_quantizer=None)


def _snr_rel_strategy():
    return _SnrRelativeStrategy(quantizer=None, high_quantizer=None)


def _iqe_strategy():
    return _IqeStrategy(quantizer=None, high_quantizer=None)


def _iqe_rel_strategy():
    return _IqeRelativeStrategy(quantizer=None, high_quantizer=None)


def _kld_strategy():
    return _KldGradientStrategy(quantizer=None, high_quantizer=None)


def test_kld_strategy_combine_sums_alignment():
    """KLD group aggregation sums per-member alignment numerators."""
    combined = _kld_strategy().combine_stats([{"alignment": 1.5}, {"alignment": -0.5}, {"alignment": 2.0}])

    assert combined["alignment"] == pytest.approx(3.0)


def test_kld_strategy_score_negates_and_normalizes_by_numel():
    """KLD scalar score is ``-alignment / (numel/1e6)`` (lower == more sensitive)."""
    assert _kld_strategy().score({"alignment": 4.0}, numel=2_000_000) == pytest.approx(-2.0)


def test_snr_strategy_combine_sums_squared_norms():
    """SNR group aggregation sums signal_sq / noise_sq across members."""
    combined = _snr_strategy().combine_stats(
        [
            {"signal_sq": 4.0, "noise_sq": 1.0},
            {"signal_sq": 9.0, "noise_sq": 1.0},
            {"signal_sq": 16.0, "noise_sq": 2.0},
        ]
    )

    assert combined["signal_sq"] == pytest.approx(29.0)
    assert combined["noise_sq"] == pytest.approx(4.0)


def test_snr_strategy_score_uses_log_of_ratio():
    """SNR scalar score is ``10*log10(signal_sq/noise_sq)``."""
    assert _snr_strategy().score({"signal_sq": 100.0, "noise_sq": 1.0}, numel=1) == pytest.approx(
        10 * math.log10(100.0)
    )


def test_snr_relative_strategy_combine_sums_high_noise():
    """SNR_RELATIVE group aggregation additionally sums the high-precision noise_sq."""
    combined = _snr_rel_strategy().combine_stats(
        [
            {"signal_sq": 4.0, "noise_sq": 1.0, "high_noise_sq": 0.25},
            {"signal_sq": 9.0, "noise_sq": 1.0, "high_noise_sq": 0.5},
        ]
    )

    assert combined["signal_sq"] == pytest.approx(13.0)
    assert combined["noise_sq"] == pytest.approx(2.0)
    assert combined["high_noise_sq"] == pytest.approx(0.75)


def test_snr_relative_strategy_score_subtracts_high_db():
    """SNR_RELATIVE scalar score is ``snr_low - snr_high``."""
    score = _snr_rel_strategy().score({"signal_sq": 100.0, "noise_sq": 1.0, "high_noise_sq": 0.01}, numel=1)

    expected = 10 * math.log10(100.0 / 1.0) - 10 * math.log10(100.0 / 0.01)
    assert score == pytest.approx(expected)


def test_iqe_strategy_combine_takes_max_of_raw():
    """IQE group aggregation takes the ``max`` of per-member max-row MSEs."""
    combined = _iqe_strategy().combine_stats([{"iqe_raw": 0.2}, {"iqe_raw": 0.5}, {"iqe_raw": 0.1}])

    assert combined["iqe_raw"] == pytest.approx(0.5)


def test_iqe_strategy_score_inverts_max_row_mse():
    """IQE scalar score is the inverse of ``max_row(mean_last((x-y)^2))``."""
    assert _iqe_strategy().score({"iqe_raw": 0.25}, numel=1) == pytest.approx(1.0 / (0.25 + 1e-12))


def test_iqe_relative_strategy_combine_takes_max_per_field():
    """IQE_RELATIVE group aggregation independently maxes low and high iqe_raw."""
    combined = _iqe_rel_strategy().combine_stats(
        [
            {"iqe_raw": 0.2, "high_iqe_raw": 0.01},
            {"iqe_raw": 0.5, "high_iqe_raw": 0.02},
            {"iqe_raw": 0.1, "high_iqe_raw": 0.05},
        ]
    )

    assert combined["iqe_raw"] == pytest.approx(0.5)
    assert combined["high_iqe_raw"] == pytest.approx(0.05)


def test_iqe_relative_strategy_score_divides_low_score_by_high_score():
    """IQE_RELATIVE scalar score is ``score_low / score_high`` (equivalently ``high_raw/low_raw``)."""
    score = _iqe_rel_strategy().score({"iqe_raw": 0.5, "high_iqe_raw": 0.1}, numel=1)

    # (1/0.5) / (1/0.1) == 0.2 (ignoring eps)
    assert score == pytest.approx(0.2, rel=1e-6)


# ---------------------------------------------------------------------------
# Strategy ``_aggregate`` (shared QKV grouping logic)
# ---------------------------------------------------------------------------


def test_aggregate_makes_singleton_units_when_no_groups():
    """With an empty ``qkv_groups``, every module becomes its own selection unit."""
    module_numels = {"a": 4, "b": 6}
    module_stats = {"a": {"signal_sq": 100.0, "noise_sq": 1.0}, "b": {"signal_sq": 100.0, "noise_sq": 0.01}}

    unit_numels, unit_scores = _snr_strategy()._aggregate(module_numels, module_stats, qkv_groups=())

    assert unit_numels == {("a",): 4, ("b",): 6}
    assert unit_scores[("a",)] == pytest.approx(10 * math.log10(100.0 / 1.0))
    assert unit_scores[("b",)] == pytest.approx(10 * math.log10(100.0 / 0.01))


def test_aggregate_collapses_qkv_groups_into_one_unit():
    """A QKV group becomes a single tuple-keyed unit; non-group modules stay singletons."""
    module_numels = {"q": 1, "k": 1, "v": 1, "down": 100}
    module_stats = {
        "q": {"signal_sq": 50.0, "noise_sq": 1.0},
        "k": {"signal_sq": 50.0, "noise_sq": 1.0},
        "v": {"signal_sq": 50.0, "noise_sq": 1.0},
        "down": {"signal_sq": 100.0, "noise_sq": 1.0},
    }

    unit_numels, unit_scores = _snr_strategy()._aggregate(module_numels, module_stats, qkv_groups=[("q", "k", "v")])

    assert unit_numels == {("q", "k", "v"): 3, ("down",): 100}
    # Group SNR = 10*log10((50+50+50)/(1+1+1)) = 10*log10(50).
    assert unit_scores[("q", "k", "v")] == pytest.approx(10 * math.log10(50.0))
    assert unit_scores[("down",)] == pytest.approx(10 * math.log10(100.0))


def test_aggregate_skips_groups_with_missing_members():
    """A QKV group is skipped if any member is absent from ``module_stats``."""
    module_numels = {"q": 1, "k": 1}
    module_stats = {"q": {"iqe_raw": 0.5}, "k": {"iqe_raw": 0.5}}

    unit_numels, unit_scores = _iqe_strategy()._aggregate(module_numels, module_stats, qkv_groups=[("q", "k", "v")])

    assert set(unit_numels) == {("q",), ("k",)}
    assert set(unit_scores) == {("q",), ("k",)}


# ---------------------------------------------------------------------------
# SNR/IQE strategy end-to-end (compute_module_stats + aggregation)
# ---------------------------------------------------------------------------


def test_snr_iqe_strategy_compute_unit_scores_returns_unit_keyed_scalars(input_model):
    """``compute_unit_scores`` returns ``(unit_numels, unit_scores)`` with scalar scores."""
    model = input_model.load_model()
    quantizer = WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False)
    high_quantizer = WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True)
    strategy = _IqeStrategy(quantizer, high_quantizer)

    unit_numels, unit_scores = strategy.compute_unit_scores(input_model, _wrap(model), "cpu", qkv_groups=())

    assert unit_numels.keys() == unit_scores.keys()
    # All units are singletons (no qkv_groups passed).
    assert all(isinstance(unit, tuple) and len(unit) == 1 for unit in unit_numels)
    assert all(isinstance(score, float) for score in unit_scores.values())


def _qkv_group_score(input_model, strategy_cls, fused_scorer):
    model = input_model.load_model()
    quantizer = WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False)
    high_quantizer = WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True)
    strategy = strategy_cls(quantizer, high_quantizer)

    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"

    _, unit_scores = strategy.compute_unit_scores(
        input_model, _wrap(model), "cpu", qkv_groups=[(q_proj, k_proj, v_proj)]
    )

    attn = model.model.layers[0].self_attn
    fused = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
    expected = fused_scorer(fused, quantizer.fake_quantize(fused))

    return unit_scores[(q_proj, k_proj, v_proj)], expected


def _snr_db(x, y, eps=1e-12):
    diff = (x - y).float()
    signal_sq = float(x.float().pow(2).sum().item())
    noise_sq = float(diff.pow(2).sum().item())
    return 10 * math.log10(max(signal_sq, eps * eps) / max(noise_sq, eps * eps))


def _iqe_score(x, y, eps=1e-12):
    diff = (x - y).float()
    raw = float(diff.pow(2).mean(dim=-1).max().item())
    return 1.0 / (raw + eps)


def test_snr_qkv_aggregation_equals_scoring_fused_tensor(input_model):
    """Aggregating per-member SNR stats equals running SNR on ``cat([Q,K,V])``."""
    aggregated, expected = _qkv_group_score(input_model, _SnrStrategy, _snr_db)

    assert aggregated == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_iqe_qkv_aggregation_equals_scoring_fused_tensor(input_model):
    """Aggregating per-member IQE stats equals running IQE on ``cat([Q,K,V])``."""
    aggregated, expected = _qkv_group_score(input_model, _IqeStrategy, _iqe_score)

    assert aggregated == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_snr_relative_qkv_aggregation_equals_scoring_fused_tensor(input_model):
    """SNR_RELATIVE aggregation equals scoring ``cat([Q,K,V])`` (Frobenius decomposition)."""
    model = input_model.load_model()
    low_q = WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False)
    high_q = WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True)
    strategy = _SnrRelativeStrategy(low_q, high_q)

    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"

    _, unit_scores = strategy.compute_unit_scores(
        input_model, _wrap(model), "cpu", qkv_groups=[(q_proj, k_proj, v_proj)]
    )

    attn = model.model.layers[0].self_attn
    fused = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
    # Score on fused tensor: SNR_low - SNR_high (db).
    expected = _snr_db(fused, low_q.fake_quantize(fused)) - _snr_db(fused, high_q.fake_quantize(fused))

    assert unit_scores[(q_proj, k_proj, v_proj)] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_iqe_relative_qkv_aggregation_equals_scoring_fused_tensor(input_model):
    """IQE_RELATIVE aggregation equals scoring ``cat([Q,K,V])`` (ratio of fused row-max MSEs)."""
    model = input_model.load_model()
    low_q = WeightQuantizer(bits=PrecisionBits.BITS4, group_size=16, symmetric=False)
    high_q = WeightQuantizer(bits=PrecisionBits.BITS8, group_size=16, symmetric=True)
    strategy = _IqeRelativeStrategy(low_q, high_q)

    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"

    _, unit_scores = strategy.compute_unit_scores(
        input_model, _wrap(model), "cpu", qkv_groups=[(q_proj, k_proj, v_proj)]
    )

    attn = model.model.layers[0].self_attn
    fused = torch.cat([attn.q_proj.weight, attn.k_proj.weight, attn.v_proj.weight], dim=0)
    # Score on fused tensor: iqe(low) / iqe(high) = (high_raw + eps) / (low_raw + eps).
    expected = _iqe_score(fused, low_q.fake_quantize(fused)) / _iqe_score(fused, high_q.fake_quantize(fused))

    assert unit_scores[(q_proj, k_proj, v_proj)] == pytest.approx(expected, rel=1e-5, abs=1e-6)


def test_kld_strategy_combine_stats_sums_alignments_and_scales_by_total_numel():
    """KLD score: aggregated unit score = -sum(alignment_i) / (sum(numel_i)/1e6)."""
    strategy = _KldGradientStrategy(quantizer=None, high_quantizer=None)

    module_numels = {"q": 200, "k": 200, "v": 200, "down": 1_000_000}
    module_stats = {
        "q": {"alignment": -1.5},
        "k": {"alignment": -2.5},
        "v": {"alignment": -4.0},
        "down": {"alignment": -100.0},
    }

    unit_numels, unit_scores = strategy._aggregate(module_numels, module_stats, qkv_groups=[("q", "k", "v")])

    assert unit_numels[("q", "k", "v")] == 600
    # Fused score = -(-1.5 + -2.5 + -4.0) / (600 / 1e6) = 8.0 / 6e-4
    assert unit_scores[("q", "k", "v")] == pytest.approx(8.0 / (600 / 1e6))
    assert unit_scores[("down",)] == pytest.approx(100.0 / (1_000_000 / 1e6))


# ---------------------------------------------------------------------------
# KLD strategy: numerical equivalence across memory modes
# ---------------------------------------------------------------------------


def _kld_unit_scores(model, memory_mode, quantizer, high_quantizer, device="cpu"):
    return _KldGradientStrategy(quantizer, high_quantizer, memory_mode=memory_mode).compute_unit_scores(
        None, _wrap(model), device, qkv_groups=()
    )


def test_kld_strategy_low_memory_matches_legacy_grad_accum(monkeypatch):
    """LOW_MEMORY KLD scoring is numerically equivalent to the legacy gradient-accumulation path."""
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    expected_numels, expected_scores = get_legacy_kld_scores(
        deepcopy(model), data, quantizer, high_quantizer, device="cpu"
    )
    actual_numels, actual_scores = _kld_unit_scores(
        deepcopy(model), KldMemoryMode.LOW_MEMORY, quantizer, high_quantizer
    )

    assert actual_numels == expected_numels
    assert_scores_close(actual_scores, expected_scores)


def test_kld_strategy_full_matches_legacy_grad_accum(monkeypatch):
    """FULL KLD scoring is numerically equivalent to the legacy gradient-accumulation path."""
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    expected_numels, expected_scores = get_legacy_kld_scores(
        deepcopy(model), data, quantizer, high_quantizer, device="cpu"
    )
    actual_numels, actual_scores = _kld_unit_scores(deepcopy(model), KldMemoryMode.FULL, quantizer, high_quantizer)

    assert actual_numels == expected_numels
    assert_scores_close(actual_scores, expected_scores)


def test_kld_strategy_offload_matches_low_memory(monkeypatch):
    """OFFLOAD mode produces the same scores as LOW_MEMORY (only differs in where tensors live)."""
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    low_memory_numels, low_memory_scores = _kld_unit_scores(
        deepcopy(model), KldMemoryMode.LOW_MEMORY, quantizer, high_quantizer
    )
    offload_numels, offload_scores = _kld_unit_scores(deepcopy(model), KldMemoryMode.OFFLOAD, quantizer, high_quantizer)

    assert offload_numels == low_memory_numels
    assert_scores_close(offload_scores, low_memory_scores)


# ---------------------------------------------------------------------------
# KLD strategy: memory mode resolution
# ---------------------------------------------------------------------------


def test_kld_strategy_auto_resolves_to_full_on_cpu():
    """AUTO mode on CPU falls back to FULL (no GPU memory budget to worry about)."""
    resolved = _KldGradientStrategy.resolve_memory_mode(
        KldGradientTestModel(), device="cpu", memory_mode=KldMemoryMode.AUTO
    )

    assert resolved == KldMemoryMode.FULL


def test_kld_strategy_auto_passthrough_when_not_auto():
    """Explicit modes are returned unchanged by the resolver."""
    for mode in (KldMemoryMode.FULL, KldMemoryMode.MULTI_GPU, KldMemoryMode.LOW_MEMORY, KldMemoryMode.OFFLOAD):
        assert _KldGradientStrategy.resolve_memory_mode(KldGradientTestModel(), device="cuda", memory_mode=mode) == mode


def test_kld_strategy_auto_falls_back_to_offload_when_cuda_memory_query_fails(monkeypatch):
    """AUTO mode chooses OFFLOAD when CUDA free memory cannot be queried safely."""

    def raise_cuda_error(_device):
        raise RuntimeError("CUDA memory query failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", raise_cuda_error)

    resolved = _KldGradientStrategy.resolve_memory_mode(
        KldGradientTestModel(), device="cuda", memory_mode=KldMemoryMode.AUTO
    )

    assert resolved == KldMemoryMode.OFFLOAD


def test_kld_strategy_auto_picks_multi_gpu_when_full_fits_across_gpus(monkeypatch):
    """AUTO picks MULTI_GPU when FULL does not fit on one GPU but fits across all visible GPUs."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # 400 bytes per GPU (single_budget=340) fails FULL on a single GPU but the combined two-GPU
    # budget (680) accommodates FULL, so MULTI_GPU should win.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (400, 400))

    resolved = _KldGradientStrategy.resolve_memory_mode(
        KldGradientTestModel(), device="cuda", memory_mode=KldMemoryMode.AUTO
    )

    assert resolved == KldMemoryMode.MULTI_GPU


def test_kld_strategy_auto_does_not_pick_multi_gpu_on_single_gpu(monkeypatch):
    """AUTO never picks MULTI_GPU when only one CUDA device is visible."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (400, 400))

    resolved = _KldGradientStrategy.resolve_memory_mode(
        KldGradientTestModel(), device="cuda", memory_mode=KldMemoryMode.AUTO
    )

    assert resolved != KldMemoryMode.MULTI_GPU


def test_kld_strategy_explicit_multi_gpu_falls_back_without_multiple_cuda_devices(monkeypatch):
    """Explicit MULTI_GPU falls back before trying CUDA when fewer than two CUDA devices are visible."""
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    warnings = []
    monkeypatch.setattr(
        smp_module.logger,
        "warning",
        lambda message, *args: warnings.append(message % args if args else message),
    )

    unit_numels, unit_scores = _kld_unit_scores(
        KldGradientTestModel(), KldMemoryMode.MULTI_GPU, quantizer, high_quantizer
    )

    assert unit_numels
    assert unit_scores
    assert any("requires at least two visible CUDA devices" in warning for warning in warnings)


def test_kld_strategy_multi_gpu_uses_constrained_device_map(monkeypatch):
    """MULTI_GPU passes constrained per-GPU max_memory to Accelerate device-map inference."""
    captured = {}
    fake_accelerate = ModuleType("accelerate")

    def fake_infer_auto_device_map(_model, max_memory, no_split_module_classes):
        captured["max_memory"] = max_memory
        captured["no_split_module_classes"] = no_split_module_classes
        return {"": 0}

    def fake_dispatch_model(model, device_map):
        captured.setdefault("device_maps", []).append(device_map)
        return model

    fake_accelerate.infer_auto_device_map = fake_infer_auto_device_map
    fake_accelerate.dispatch_model = fake_dispatch_model
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "accelerate" else original_find_spec(name),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (1000, 1000))
    patch_kld_calibration_data(monkeypatch, [])
    monkeypatch.setattr(smp_module, "replace_matching_submodules", lambda *_args, **_kwargs: None)
    quantizer, high_quantizer = get_kld_gradient_quantizers()

    _kld_unit_scores(KldGradientTestModel(), KldMemoryMode.MULTI_GPU, quantizer, high_quantizer, device="cuda")

    assert captured["max_memory"] == {0: 222, 1: 222}
    assert all(memory < 850 for memory in captured["max_memory"].values())
    assert captured["device_maps"] == [{"": 0}, {"": 0}]


def test_kld_strategy_multi_gpu_logs_per_layer_device_counts(monkeypatch):
    """MULTI_GPU diagnostic log reports decoder-layer counts per device, not raw map entries."""
    fake_accelerate = ModuleType("accelerate")

    def fake_infer_auto_device_map(_model, max_memory, no_split_module_classes):
        return {
            "model.layers.0": 0,
            "model.layers.0.mlp.down_proj": 1,  # to be coalesced back to device 0
            "model.layers.1": 0,
            "model.layers.2": 1,
            "model.embed_tokens": 0,
            "lm_head": 1,
        }

    def fake_dispatch_model(model, device_map):
        return model

    fake_accelerate.infer_auto_device_map = fake_infer_auto_device_map
    fake_accelerate.dispatch_model = fake_dispatch_model
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "accelerate" else original_find_spec(name),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (1000, 1000))
    patch_kld_calibration_data(monkeypatch, [])
    monkeypatch.setattr(smp_module, "replace_matching_submodules", lambda *_args, **_kwargs: None)
    quantizer, high_quantizer = get_kld_gradient_quantizers()

    import logging as _logging

    captured_logs: list[str] = []

    class _ListHandler(_logging.Handler):
        def emit(self, record):
            captured_logs.append(record.getMessage())

    handler = _ListHandler(level=_logging.INFO)
    smp_module.logger.addHandler(handler)
    previous_level = smp_module.logger.level
    smp_module.logger.setLevel(_logging.INFO)
    try:
        _kld_unit_scores(KldGradientTestModel(), KldMemoryMode.MULTI_GPU, quantizer, high_quantizer, device="cuda")
    finally:
        smp_module.logger.removeHandler(handler)
        smp_module.logger.setLevel(previous_level)

    device_map_logs = [msg for msg in captured_logs if "kld_memory_mode=multi_gpu device_map" in msg]
    assert device_map_logs, f"expected an info log describing the multi_gpu device map; got {captured_logs!r}"
    log = device_map_logs[-1]
    assert "3 decoder layers" in log
    assert "'0': 2" in log or "0: 2" in log
    assert "'1': 1" in log or "1: 1" in log


def test_kld_strategy_multi_gpu_coalesces_using_wrapper_layer_prefix(monkeypatch):
    """MULTI_GPU coalescing uses ``model_wrapper.get_layers`` so non-llama layouts work too."""
    fake_accelerate = ModuleType("accelerate")
    captured: dict = {}

    def fake_infer_auto_device_map(_model, max_memory, no_split_module_classes):
        # Decoder layers live under ``transformer.h`` for gpt-neox-style layouts. A submodule
        # of layer 0 lands on device 1; coalescing must pull it back to device 0.
        return {
            "transformer.h.0": 0,
            "transformer.h.0.mlp.down_proj": 1,
            "transformer.h.1": 1,
            "lm_head": 1,
        }

    def fake_dispatch_model(model, device_map):
        captured.setdefault("device_maps", []).append(device_map)
        return model

    fake_accelerate.infer_auto_device_map = fake_infer_auto_device_map
    fake_accelerate.dispatch_model = fake_dispatch_model
    monkeypatch.setitem(sys.modules, "accelerate", fake_accelerate)
    original_find_spec = importlib.util.find_spec
    monkeypatch.setattr(
        importlib.util,
        "find_spec",
        lambda name: object() if name == "accelerate" else original_find_spec(name),
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (1000, 1000))
    patch_kld_calibration_data(monkeypatch, [])
    monkeypatch.setattr(smp_module, "replace_matching_submodules", lambda *_args, **_kwargs: None)
    quantizer, high_quantizer = get_kld_gradient_quantizers()

    strategy = _KldGradientStrategy(quantizer, high_quantizer, memory_mode=KldMemoryMode.MULTI_GPU)
    strategy.compute_unit_scores(None, _wrap(KldGradientTestModel(), layer_prefix="transformer.h"), "cuda", ())

    # Coalescing must have run: layer 0's submodule on device 1 is pulled back to layer 0's device 0.
    final_map = captured["device_maps"][-1]
    assert final_map["transformer.h.0.mlp.down_proj"] == 0
    assert final_map["transformer.h.0"] == 0
    assert final_map["transformer.h.1"] == 1


def test_kld_strategy_memory_modes_keep_same_mixed_precision_config(input_model, monkeypatch):
    """All KLD memory modes yield identical ``mixed_precision_info`` given the same scores.

    Memory mode is a runtime-only knob; once scoring is fixed, the produced config must not
    depend on which path was taken. Also asserts each mode is plumbed through to the strategy.
    """
    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"
    down_proj = "model.layers.0.mlp.down_proj"
    unit_numels = {(q_proj, k_proj, v_proj): 3, (down_proj,): 100}
    unit_scores = {(q_proj, k_proj, v_proj): 20.0, (down_proj,): -500.0}
    seen_memory_modes = []

    def fake_compute_unit_scores(self, *_args, **_kwargs):
        seen_memory_modes.append(self.memory_mode)
        return unit_numels, unit_scores

    monkeypatch.setattr(_KldGradientStrategy, "compute_unit_scores", fake_compute_unit_scores)
    model_wrapper = ModelWrapper.from_model(input_model.load_model())

    modes = [KldMemoryMode.FULL, KldMemoryMode.LOW_MEMORY, KldMemoryMode.OFFLOAD]
    configs = [
        SelectiveMixedPrecision.get_scored_config(
            input_model,
            model_wrapper,
            SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
            PrecisionBits.BITS4,
            16,
            False,
            PrecisionBits.BITS8,
            16,
            True,
            0.98,
            mode,
        )
        for mode in modes
    ]

    assert configs[0] == configs[1] == configs[2]
    assert seen_memory_modes == modes
