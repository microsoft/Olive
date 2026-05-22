# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.util
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
from olive.passes.pytorch.quant_utils import _quant_config_rank, get_qkv_quantization_groups, prepare_model
from olive.passes.pytorch.selective_mixed_precision import SelectiveMixedPrecision
from olive.passes.pytorch.train_utils import kl_div_loss
from test.utils import get_tiny_phi3


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

    scores = {}
    with torch.no_grad():
        for module_name, layer in quantized_layers.items():
            grad = grad_accum[module_name] / len(data)
            high_weight = high_quantizer.fake_quantize(model.get_submodule(module_name).weight.data)
            param_size_m = module_numels[module_name] / 1e6
            scores[module_name] = -((grad * (layer.weight.data - high_weight)).sum().item() / param_size_m)

    return module_numels, scores


def patch_kld_calibration_data(monkeypatch, data):
    monkeypatch.setattr(
        "olive.passes.pytorch.selective_mixed_precision.get_calibration_dataset",
        lambda *_args, **_kwargs: data,
    )


def assert_scores_close(actual_scores, expected_scores):
    assert actual_scores.keys() == expected_scores.keys()
    for module_name, expected_score in expected_scores.items():
        assert actual_scores[module_name] == pytest.approx(expected_score, rel=1e-6, abs=1e-6)


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


@pytest.mark.parametrize(
    ("algorithm", "expected_layer_indices", "include_qkv"),
    [
        ("k_quant_down", [0, 3, 6, 7], False),  # first 1/8, every 3rd, and last 1/8
        ("k_quant_mixed", [0, 3, 6, 7], True),
        ("k_quant_last", [], False),
    ],
)
def test_selective_mixed_precision_k_quant(algorithm, expected_layer_indices, include_qkv, input_model, tmp_path):
    """End-to-end: rule-based k_quant_* algorithms write the expected mixed_precision_info.

    Verifies that each k_quant variant promotes the correct subset of layers (first 1/8,
    every 3rd, last 1/8) and that ``k_quant_mixed`` additionally promotes Q/K/V together,
    while ``k_quant_last`` (lm_head only) leaves all transformer layers untouched.
    """
    config = {"algorithm": algorithm}
    p = create_pass_from_dict(SelectiveMixedPrecision, config, disable_search=True)

    output_model = p.run(input_model, str(tmp_path))

    # Check that mixed_precision_info was added
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


def test_selective_mixed_precision_scored_keeps_qkv_same_precision(input_model):
    """Score-based selection must promote Q/K/V as a single group (separate-projection model).

    Even though only k_proj has the low/sensitive score, q_proj/v_proj must also be promoted
    so the three attention input projections share the same precision (required by fused QKV
    kernels downstream).
    """
    model_wrapper = ModelWrapper.from_model(input_model.load_model())
    q_proj = "model.layers.0.self_attn.q_proj"
    k_proj = "model.layers.0.self_attn.k_proj"
    v_proj = "model.layers.0.self_attn.v_proj"
    down_proj = "model.layers.0.mlp.down_proj"
    module_numels = {q_proj: 1, k_proj: 1, v_proj: 1, down_proj: 100}
    module_scores = {q_proj: 10.0, k_proj: 1.0, v_proj: 10.0, down_proj: 5.0}

    overrides, high_precision_numels = SelectiveMixedPrecision.get_overrides_from_scores(
        model_wrapper,
        module_numels,
        module_scores,
        ratio=0.98,
        high_override_config={"bits": PrecisionBits.BITS8},
    )

    assert overrides == {
        q_proj: {"bits": PrecisionBits.BITS8},
        k_proj: {"bits": PrecisionBits.BITS8},
        v_proj: {"bits": PrecisionBits.BITS8},
    }
    assert high_precision_numels == 3


def test_selective_mixed_precision_scored_ignores_single_packed_qkv():
    """A packed single qkv_proj (phi3 before unpacking) is not treated as a QKV group.

    Grouping only applies when Q/K/V are distinct modules; a fused single projection has
    nothing to co-promote and must be skipped.
    """
    model_wrapper = ModelWrapper.from_model(get_tiny_phi3().load_model())

    assert not get_qkv_quantization_groups(model_wrapper, {"model.layers.0.self_attn.qkv_proj"})


def test_selective_mixed_precision_scored_groups_unpacked_qkv():
    """After ``maybe_unpack_qkv()``, the unpacked Q/K/V submodules form a single group.

    Confirms that phi3-style models still get correct QKV co-promotion once the packed
    qkv_proj has been split into ``qkv_proj.{q,k,v}_proj`` children.
    """
    model_wrapper = ModelWrapper.from_model(get_tiny_phi3().load_model())
    model_wrapper.maybe_unpack_qkv()
    q_proj = "model.layers.0.self_attn.qkv_proj.q_proj"
    k_proj = "model.layers.0.self_attn.qkv_proj.k_proj"
    v_proj = "model.layers.0.self_attn.qkv_proj.v_proj"
    down_proj = "model.layers.0.mlp.down_proj"
    module_numels = {q_proj: 1, k_proj: 1, v_proj: 1, down_proj: 100}
    module_scores = {q_proj: 10.0, k_proj: 1.0, v_proj: 10.0, down_proj: 5.0}

    overrides, high_precision_numels = SelectiveMixedPrecision.get_overrides_from_scores(
        model_wrapper,
        module_numels,
        module_scores,
        ratio=0.98,
        high_override_config={"bits": PrecisionBits.BITS8},
    )

    assert overrides == {
        q_proj: {"bits": PrecisionBits.BITS8},
        k_proj: {"bits": PrecisionBits.BITS8},
        v_proj: {"bits": PrecisionBits.BITS8},
    }
    assert high_precision_numels == 3


def test_quant_config_promotes_user_override_conflicts_for_qkv(input_model):
    """``normalize_qkv_quant_config`` promotes the most-precise config across the QKV group.

    When k_proj/v_proj are int8/sym/g16 but q_proj is left at int4, all three must end up
    int8/sym/g16 (highest bits wins) so the fused kernel sees a single shared config.
    """
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "mixed_precision_info": {
                "default": {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False},
                "overrides": {
                    "model.layers.0.self_attn.k_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                    "model.layers.0.self_attn.v_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                },
            }
        },
    )
    config = SimpleNamespace(
        bits=PrecisionBits.BITS4,
        sym=False,
        group_size=16,
        lm_head=False,
        overrides={"model.layers.0.self_attn.q_proj": {"bits": PrecisionBits.BITS4}},
    )

    wrapper, qcfg, _ = prepare_model(model, config)

    qkv_qargs = [qcfg.get_qlinear_init_args(f"model.layers.0.self_attn.{name}_proj") for name in ["q", "k", "v"]]
    assert qkv_qargs == [
        {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16},
        {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16},
        {"bits": PrecisionBits.BITS8, "symmetric": True, "group_size": 16},
    ]
    assert [
        getattr(wrapper.model.model.layers[0].self_attn, f"{name}_proj").quant_info.quantizer.bits
        for name in ["q", "k", "v"]
    ] == [PrecisionBits.BITS8, PrecisionBits.BITS8, PrecisionBits.BITS8]


def test_quant_config_rank_prefers_bits_then_smaller_positive_group_size():
    """Unit test for ``_quant_config_rank`` ordering used to promote QKV groups.

    Higher ``bits`` wins; among equal bits, smaller positive ``group_size`` wins; per-channel
    wins over per-tensor; ``symmetric`` does not affect the rank.
    """
    symmetric_qargs = {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": True}
    asymmetric_qargs = {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False}
    group_size_qargs = [
        {"bits": PrecisionBits.BITS4, "group_size": 128, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": 32, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": -1, "symmetric": True},
        {"bits": PrecisionBits.BITS4, "group_size": 0, "symmetric": True},
    ]
    higher_bit_qargs = {"bits": PrecisionBits.BITS8, "group_size": 128, "symmetric": True}

    assert _quant_config_rank(symmetric_qargs) == _quant_config_rank(asymmetric_qargs)
    assert max(group_size_qargs, key=_quant_config_rank) == group_size_qargs[1]
    assert max(group_size_qargs[2:], key=_quant_config_rank) == group_size_qargs[2]
    assert max([*group_size_qargs, higher_bit_qargs], key=_quant_config_rank) == higher_bit_qargs


def test_quant_config_ignores_excluded_qkv_overrides_when_normalizing(input_model):
    """With ``exclude_attn_inputs=True``, excluded Q/K do not pull V into a higher precision.

    Q and K are excluded from quantization, so their high-bit overrides must be dropped and
    V must keep the default int4 config (no group promotion across excluded members).
    """
    model = HfModelHandler(
        input_model.model_path,
        model_attributes={
            "mixed_precision_info": {
                "default": {"bits": PrecisionBits.BITS4, "group_size": 16, "symmetric": False},
                "overrides": {
                    "model.layers.0.self_attn.q_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                    "model.layers.0.self_attn.k_proj": {
                        "bits": PrecisionBits.BITS8,
                        "group_size": 16,
                        "symmetric": True,
                    },
                },
            }
        },
    )
    config = SimpleNamespace(
        bits=PrecisionBits.BITS4,
        sym=False,
        group_size=16,
        lm_head=False,
        overrides=None,
    )

    wrapper, qcfg, _ = prepare_model(model, config, exclude_attn_inputs=True)
    attention = wrapper.model.model.layers[0].self_attn

    assert not hasattr(attention.q_proj, "quant_info")
    assert not hasattr(attention.k_proj, "quant_info")
    assert qcfg.get_qlinear_init_args("model.layers.0.self_attn.v_proj") == {
        "bits": PrecisionBits.BITS4,
        "symmetric": False,
        "group_size": 16,
    }
    assert attention.v_proj.quant_info.quantizer.bits == PrecisionBits.BITS4
    assert "model.layers.0.self_attn.q_proj" not in qcfg.overrides
    assert "model.layers.0.self_attn.k_proj" not in qcfg.overrides


def test_selective_mixed_precision_kld_low_memory_matches_legacy_grad_accum(monkeypatch):
    """LOW_MEMORY KLD scoring is numerically equivalent to the legacy gradient-accumulation path.

    Guards the chunked/low-memory implementation against drift from the reference scores.
    """
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    expected_numels, expected_scores = get_legacy_kld_scores(
        deepcopy(model), data, quantizer, high_quantizer, device="cpu"
    )
    actual_numels, actual_scores = SelectiveMixedPrecision.get_kld_scores(
        None,
        deepcopy(model),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cpu",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY,
    )

    assert actual_numels == expected_numels
    assert_scores_close(actual_scores, expected_scores)


def test_selective_mixed_precision_kld_full_matches_legacy_grad_accum(monkeypatch):
    """FULL KLD scoring is numerically equivalent to the legacy gradient-accumulation path.

    Pins the fast/full path to the same scores as the reference implementation.
    """
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    expected_numels, expected_scores = get_legacy_kld_scores(
        deepcopy(model), data, quantizer, high_quantizer, device="cpu"
    )
    actual_numels, actual_scores = SelectiveMixedPrecision.get_kld_scores(
        None,
        deepcopy(model),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cpu",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.FULL,
    )

    assert actual_numels == expected_numels
    assert_scores_close(actual_scores, expected_scores)


def test_selective_mixed_precision_kld_auto_resolves_to_full_on_cpu():
    """AUTO mode on CPU falls back to FULL (no GPU memory budget to worry about)."""
    model = KldGradientTestModel()

    resolved = SelectiveMixedPrecision.resolve_kld_memory_mode(
        model, device="cpu", kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.AUTO
    )

    assert resolved == SelectiveMixedPrecision.KldMemoryMode.FULL


def test_selective_mixed_precision_kld_auto_passthrough_when_not_auto():
    """Explicit modes (FULL/LOW_MEMORY/OFFLOAD) are returned unchanged by the resolver.

    Only AUTO triggers heuristic selection; user-pinned modes must pass through.
    """
    model = KldGradientTestModel()

    for mode in (
        SelectiveMixedPrecision.KldMemoryMode.FULL,
        SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU,
        SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY,
        SelectiveMixedPrecision.KldMemoryMode.OFFLOAD,
    ):
        assert SelectiveMixedPrecision.resolve_kld_memory_mode(model, device="cuda", kld_memory_mode=mode) == mode


def test_selective_mixed_precision_kld_auto_falls_back_to_offload_when_cuda_memory_query_fails(monkeypatch):
    """AUTO mode chooses OFFLOAD when CUDA free memory cannot be queried safely."""
    model = KldGradientTestModel()

    def raise_cuda_error(_device):
        raise RuntimeError("CUDA memory query failed")

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "mem_get_info", raise_cuda_error)

    resolved = SelectiveMixedPrecision.resolve_kld_memory_mode(
        model, device="cuda", kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.AUTO
    )

    assert resolved == SelectiveMixedPrecision.KldMemoryMode.OFFLOAD


def test_selective_mixed_precision_kld_auto_picks_multi_gpu_when_full_fits_across_gpus(monkeypatch):
    """AUTO picks MULTI_GPU when FULL does not fit on one GPU but fits across all visible GPUs.

    Simulates two GPUs each with a small per-device budget so that FULL fails the single-GPU
    check but the combined budget across both GPUs is sufficient.
    """
    model = KldGradientTestModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    # The tiny test model needs ~491 bytes for FULL and ~338 bytes for LOW_MEMORY. 400 bytes per GPU
    # (single_budget=340) fails the FULL single-GPU check but the combined two-GPU budget (680)
    # accommodates FULL, so MULTI_GPU should win.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (400, 400))

    resolved = SelectiveMixedPrecision.resolve_kld_memory_mode(
        model, device="cuda", kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.AUTO
    )

    assert resolved == SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU


def test_selective_mixed_precision_kld_auto_does_not_pick_multi_gpu_on_single_gpu(monkeypatch):
    """AUTO never picks MULTI_GPU when only one CUDA device is visible, even if FULL doesn't fit.

    With a single GPU the ladder must skip MULTI_GPU and pick LOW_MEMORY or OFFLOAD instead.
    """
    model = KldGradientTestModel()

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    # 400 bytes single-GPU budget (340 after safety factor): FULL won't fit, MULTI_GPU is gated
    # out by gpu_count==1, so LOW_MEMORY/OFFLOAD must win.
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda _i: (400, 400))

    resolved = SelectiveMixedPrecision.resolve_kld_memory_mode(
        model, device="cuda", kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.AUTO
    )

    assert resolved != SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU


def test_selective_mixed_precision_kld_explicit_multi_gpu_falls_back_without_multiple_cuda_devices(monkeypatch):
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

    module_numels, module_scores = SelectiveMixedPrecision.get_kld_scores(
        None,
        KldGradientTestModel(),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cpu",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU,
    )

    assert module_numels
    assert module_scores
    assert any("requires at least two visible CUDA devices" in warning for warning in warnings)


def test_selective_mixed_precision_kld_multi_gpu_uses_constrained_device_map(monkeypatch):
    """MULTI_GPU passes constrained per-GPU max_memory to Accelerate device-map inference.

    This prevents Accelerate from placing one full model copy on a GPU without leaving room for
    the second copy and the fp32 gradient accumulator used by the FULL algorithm.
    """
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

    SelectiveMixedPrecision.get_kld_scores(
        None,
        KldGradientTestModel(),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cuda",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.MULTI_GPU,
    )

    assert captured["max_memory"] == {0: 222, 1: 222}
    assert all(memory < 850 for memory in captured["max_memory"].values())
    assert captured["device_maps"] == [{"": 0}, {"": 0}]


def test_selective_mixed_precision_kld_offload_matches_low_memory(monkeypatch):
    """OFFLOAD mode produces the same scores as LOW_MEMORY (only differs in where tensors live).

    Confirms CPU-offloading does not perturb the numerics relative to the in-GPU low-memory path.
    """
    data = get_kld_gradient_test_data()
    patch_kld_calibration_data(monkeypatch, data)
    quantizer, high_quantizer = get_kld_gradient_quantizers()
    model = KldGradientTestModel()

    low_memory_numels, low_memory_scores = SelectiveMixedPrecision.get_kld_scores(
        None,
        deepcopy(model),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cpu",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY,
    )
    offload_numels, offload_scores = SelectiveMixedPrecision.get_kld_scores(
        None,
        deepcopy(model),
        SelectiveMixedPrecision.Algorithm.KLD_GRADIENT,
        quantizer,
        high_quantizer,
        device="cpu",
        kld_memory_mode=SelectiveMixedPrecision.KldMemoryMode.OFFLOAD,
    )

    assert offload_numels == low_memory_numels
    assert_scores_close(offload_scores, low_memory_scores)


def test_selective_mixed_precision_kld_memory_modes_keep_same_mixed_precision_config(input_model, monkeypatch):
    """FULL/LOW_MEMORY/OFFLOAD all yield identical ``mixed_precision_info`` given the same scores.

    Memory mode is a runtime-only knob; once scoring is mocked to a fixed result, the produced
    config must not depend on which path was taken. Also asserts each mode is actually passed
    through to ``get_kld_scores``.
    """
    module_numels = {
        "model.layers.0.self_attn.q_proj": 1,
        "model.layers.0.self_attn.k_proj": 1,
        "model.layers.0.self_attn.v_proj": 1,
        "model.layers.0.mlp.down_proj": 100,
    }
    module_scores = {
        "model.layers.0.self_attn.q_proj": 10.0,
        "model.layers.0.self_attn.k_proj": 1.0,
        "model.layers.0.self_attn.v_proj": 10.0,
        "model.layers.0.mlp.down_proj": 5.0,
    }
    seen_memory_modes = []

    def get_kld_scores(*args):
        seen_memory_modes.append(args[-1])
        return module_numels, module_scores

    monkeypatch.setattr(SelectiveMixedPrecision, "get_kld_scores", staticmethod(get_kld_scores))
    model_wrapper = ModelWrapper.from_model(input_model.load_model())

    modes = [
        SelectiveMixedPrecision.KldMemoryMode.FULL,
        SelectiveMixedPrecision.KldMemoryMode.LOW_MEMORY,
        SelectiveMixedPrecision.KldMemoryMode.OFFLOAD,
    ]
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


@pytest.mark.parametrize("algorithm", ["snr", "snr_relative", "iqe", "iqe_relative", "kld_gradient"])
def test_selective_mixed_precision_scored(algorithm, tmp_path):
    """End-to-end: every score-based algorithm produces a valid ``mixed_precision_info``.

    Runs the pass on tiny-phi3 for each scoring algorithm and checks the default/lm_head
    sections are populated correctly. ``kld_gradient`` is skipped on CPU (too slow).
    """
    if algorithm == "kld_gradient" and not torch.cuda.is_available():
        pytest.skip("Skipping kld_gradient test as it runs slow on CPU.")

    p = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": algorithm, "ratio": 0.8, "group_size": 16, "high_sym": True},
        disable_search=True,
    )

    output_model = p.run(get_tiny_phi3(), str(tmp_path))

    # Check that mixed_precision_info was added
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
