# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Tests for ``olive.common.quant.selection.iter_quant_targets``."""

from __future__ import annotations

import torch
import torch.nn as nn

from olive.common.quant.selection import iter_quant_targets


class _Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.linear = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


def _names(targets):
    return sorted(full_name for _, _, full_name in targets)


def test_default_skips_lm_head_and_embeds():
    m = _Toy()
    targets = list(iter_quant_targets(m, quantize_lm_head=False, quantize_embeds=False, quantize_moe=False))
    assert _names(targets) == ["linear"]


def test_include_lm_head_and_embeds():
    m = _Toy()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["embed_tokens", "linear", "lm_head"]


class _MultiEmbed(nn.Module):
    """A model with more than one nn.Embedding, like a legacy BERT/GPT-2 style model."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.position_embeddings = nn.Embedding(32, 8)
        self.linear = nn.Linear(8, 8, bias=False)

    def get_input_embeddings(self):
        return self.embed_tokens


def test_embeds_true_targets_only_input_embeddings_when_resolvable():
    """D: quantize_embeds=True should target ONLY get_input_embeddings(), not every nn.Embedding."""
    m = _MultiEmbed()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["embed_tokens", "linear"]


def test_embeds_false_still_skips_all_embeddings():
    """D: quantize_embeds=False must still skip ALL nn.Embedding modules (loophole prevention)."""
    m = _MultiEmbed()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=False, quantize_moe=False))
    assert _names(targets) == ["linear"]


class _NoInputEmbedsAccessor(nn.Module):
    """Synthetic fixture without get_input_embeddings -- fallback path."""

    def __init__(self):
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 8)
        self.position_embeddings = nn.Embedding(32, 8)
        self.linear = nn.Linear(8, 8, bias=False)


def test_embeds_true_falls_back_to_all_embeddings_without_accessor():
    """D fallback: when get_input_embeddings is unavailable, retain the broad behavior."""
    m = _NoInputEmbedsAccessor()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["embed_tokens", "linear", "position_embeddings"]


def test_skip_patterns_filter_by_name():
    m = _Toy()
    targets = list(
        iter_quant_targets(
            m,
            quantize_lm_head=True,
            quantize_embeds=True,
            quantize_moe=False,
            skip_patterns=["re:.*_head"],
        )
    )
    assert _names(targets) == ["embed_tokens", "linear"]


def test_extra_skip_modules_skip_by_identity():
    m = _Toy()
    targets = list(
        iter_quant_targets(
            m,
            quantize_lm_head=True,
            quantize_embeds=False,
            quantize_moe=False,
            extra_skip_modules={m.linear},
        )
    )
    assert _names(targets) == ["lm_head"]


def test_already_quantized_param_is_skipped():
    from olive.common.quant.tensor import QuantTensor

    m = _Toy()
    qt = QuantTensor.from_packed(
        qweight=torch.zeros((8, 4), dtype=torch.uint8),
        scales=torch.zeros((8, 1), dtype=torch.float32),
        qzeros=None,
        bits=4,
        group_size=8,
        symmetric=True,
        shape=(8, 8),
        dtype=torch.float32,
    )
    m.linear.weight = nn.Parameter(qt, requires_grad=False)

    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=False, quantize_moe=False))
    assert _names(targets) == ["lm_head"]


class _ExpertList(nn.Module):
    def __init__(self):
        super().__init__()
        self.experts = nn.ModuleList([nn.Linear(8, 8, bias=False) for _ in range(2)])


def test_moe_disabled_skips_submodules_under_experts(monkeypatch):
    """When ``quantize_moe=False``, every nn.Module under an experts subtree is skipped."""
    from olive.common.hf import wrapper as wrapper_mod

    class FakeLayerWrapper:
        def __init__(self, experts, name):
            self._experts = experts
            self._name = name

        def get_experts(self, return_name=True):
            return (self._experts, self._name) if return_name else self._experts

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [FakeLayerWrapper(self.model.experts, "experts")]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    m = _ExpertList()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == []


def test_moe_enabled_yields_3d_fused_params(monkeypatch):
    from olive.common.hf import wrapper as wrapper_mod

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)
            self.down_proj = nn.Parameter(torch.zeros(4, 16, 8), requires_grad=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()

    class FakeLayerWrapper:
        def __init__(self, experts, name):
            self._experts = experts
            self._name = name

        def get_experts(self, return_name=True):
            return (self._experts, self._name) if return_name else self._experts

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [FakeLayerWrapper(self.model.experts, "experts")]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    m = _Model()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    fused = sorted((full_name, tuple(module._parameters[pname].shape)) for module, pname, full_name in targets)
    assert fused == [
        ("experts.down_proj", (4, 16, 8)),
        ("experts.gate_up_proj", (4, 8, 16)),
    ]


def _install_fake_wrapper(monkeypatch, experts_by_layer):
    """Patch ModelWrapper.from_model to expose ``experts_by_layer`` (list of (module, name))."""
    from olive.common.hf import wrapper as wrapper_mod

    class FakeLayerWrapper:
        def __init__(self, experts, name):
            self._experts = experts
            self._name = name

        def get_experts(self, return_name=True):
            return (self._experts, self._name) if return_name else self._experts

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [FakeLayerWrapper(e, n) for e, n in experts_by_layer]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))


def test_moe_enabled_yields_only_3d_weights_and_skips_2d_bias(monkeypatch):
    """Regression (gpt-oss gap): 2D bias params on a fused experts module must NOT be quantized."""

    class FusedExpertsWithBias(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)
            self.gate_up_proj_bias = nn.Parameter(torch.zeros(4, 8), requires_grad=False)  # 2D bias
            self.down_proj = nn.Parameter(torch.zeros(4, 16, 8), requires_grad=False)
            self.down_proj_bias = nn.Parameter(torch.zeros(4, 16), requires_grad=False)  # 2D bias

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExpertsWithBias()

    m = _Model()
    _install_fake_wrapper(monkeypatch, [(m.experts, "experts")])

    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert _names(targets) == ["experts.down_proj", "experts.gate_up_proj"]
    # every yielded param is 3D (no 2D bias slipped in)
    assert all(module._parameters[pname].dim() == 3 for module, pname, _ in targets)


def test_shared_expert_gate_excluded_from_quantization(monkeypatch):
    """Verify ``SHARED_EXPERT_GATE``-mapped modules are excluded from quantization.

    Covers qwen2_moe/qwen3_5_moe/qwen3_next-style single-row sigmoid gates, which must be
    excluded the same way routers are, regardless of ``quantize_moe``.
    """
    from olive.common.hf import wrapper as wrapper_mod

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()
            self.shared_expert_gate = nn.Linear(8, 1, bias=False)
            self.other_linear = nn.Linear(8, 8, bias=False)

    class FakeLayerWrapper:
        def __init__(self, experts, gate):
            self._experts = experts
            self._gate = gate

        def get_experts(self, return_name=True):
            return (self._experts, "experts") if return_name else self._experts

        def get_shared_expert_gate(self, return_name=True):
            return (self._gate, "shared_expert_gate") if return_name else self._gate

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [FakeLayerWrapper(self.model.experts, self.model.shared_expert_gate)]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    m = _Model()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert _names(targets) == ["experts.gate_up_proj", "other_linear"]


def test_mamba_linear_attn_excluded_from_quantization(monkeypatch):
    """Verify ``MAMBA``-mapped modules are excluded from the generic 2D quantization walk.

    Includes qwen3_5_moe's ``linear_attn`` GatedDeltaNet block, which must be excluded
    unconditionally.
    """
    from olive.common.hf import wrapper as wrapper_mod

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)

    class LinearAttn(nn.Module):
        def __init__(self):
            super().__init__()
            self.in_proj_qkv = nn.Linear(8, 24, bias=False)
            self.out_proj = nn.Linear(8, 8, bias=False)

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.experts = FusedExperts()
            self.linear_attn = LinearAttn()
            self.other_linear = nn.Linear(8, 8, bias=False)

    class FakeLayerWrapper:
        def __init__(self, experts, mamba):
            self._experts = experts
            self._mamba = mamba

        def get_experts(self, return_name=True):
            return (self._experts, "experts") if return_name else self._experts

        def get_mamba(self, return_name=True):
            return (self._mamba, "linear_attn") if return_name else self._mamba

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [FakeLayerWrapper(self.model.experts, self.model.linear_attn)]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    m = _Model()
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert _names(targets) == ["experts.gate_up_proj", "other_linear"]


def test_modulelist_experts_moe_flag_controls_selection(monkeypatch):
    """Regression (ModuleList bug): per-expert Linears are quantized iff ``moe=True``."""
    m = _ExpertList()
    _install_fake_wrapper(monkeypatch, [(m.experts, "experts")])

    off = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(off) == []

    on = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert _names(on) == ["experts.0", "experts.1"]


def test_fail_closed_when_moe_arch_but_experts_not_discovered(monkeypatch):
    """``moe=False`` must fail closed for an MoE arch whose experts can't be resolved."""

    class _MoEConfig:
        num_local_experts = 8

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _MoEConfig()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    # Wrapper resolves no experts (unrecognized architecture).
    _install_fake_wrapper(monkeypatch, [])

    import pytest

    with pytest.raises(ValueError, match="Mixture-of-Experts"):
        list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))


def test_fail_closed_for_dbrx_shaped_nested_config(monkeypatch):
    """R2-4: DBRX-shaped nested config (``config.ffn_config.moe_num_experts``) must be detected.

    Previously ``_config_indicates_moe`` only checked a hardcoded top-level attribute tuple,
    so this nested-config MoE architecture silently slipped through the fail-closed guard.
    """

    class _FfnConfig:
        moe_num_experts = 8

    class _DbrxConfig:
        ffn_config = _FfnConfig()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DbrxConfig()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    _install_fake_wrapper(monkeypatch, [])

    import pytest

    with pytest.raises(ValueError, match="Mixture-of-Experts"):
        list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))


def test_fail_closed_for_llama4_shaped_nested_config(monkeypatch):
    """R2-4: Llama4-shaped nested config (``config.text_config.num_local_experts``) is detected."""

    class _TextConfig:
        num_local_experts = 16

    class _Llama4Config:
        text_config = _TextConfig()

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Llama4Config()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    _install_fake_wrapper(monkeypatch, [])

    import pytest

    with pytest.raises(ValueError, match="Mixture-of-Experts"):
        list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))


def test_dense_config_does_not_trigger_moe_guard(monkeypatch):
    """A plain dense config (only ``num_hidden_layers``) must not trigger the MoE fail-closed guard."""

    class _DenseConfig:
        num_hidden_layers = 12

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _DenseConfig()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    _install_fake_wrapper(monkeypatch, [])

    # Should not raise -- falls through to the plain 2D walk.
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["some_linear"]


def test_zero_experts_does_not_trigger_moe_guard(monkeypatch):
    """``num_experts = 0`` (falsy/absent MoE) must not trigger the fail-closed guard."""

    class _Config:
        num_experts = 0

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = _Config()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    _install_fake_wrapper(monkeypatch, [])

    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["some_linear"]


def test_magicmock_config_does_not_trigger_moe_guard(monkeypatch):
    """A ``MagicMock`` config must not be spuriously treated as MoE.

    Guards against `Mock` objects being truthy for every ``getattr`` -- the generic
    sub-config sweep must reject non-``int`` values (including further ``Mock`` objects).
    """
    from unittest.mock import MagicMock

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = MagicMock()
            self.some_linear = nn.Linear(8, 8, bias=False)

    m = _Model()
    _install_fake_wrapper(monkeypatch, [])

    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert _names(targets) == ["some_linear"]


def test_fail_closed_per_layer_when_some_layers_have_router_but_no_experts(monkeypatch):
    """M4: fail closed per-layer, not only when ALL layers fail to resolve experts.

    Layer 0 resolves experts fine; layer 1 has a resolvable router/gate but its experts
    subtree fails to resolve -- this must raise before any target is yielded, even though
    layer 0 succeeded. A dense layer with *no* router (layer 2) is legitimately expert-free
    (e.g. DeepSeek's ``first_k_dense_replace``) and must NOT trip the guard.
    """
    from olive.common.hf import wrapper as wrapper_mod

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)

    class _Router(nn.Module):
        pass

    class FakeLayerWrapper:
        def __init__(self, experts, router):
            self._experts = experts
            self._router = router

        def get_experts(self, return_name=True):
            return (self._experts, "experts") if return_name else self._experts

        def get_router(self, return_name=True):
            return (self._router, "gate") if return_name else self._router

    resolved_experts = FusedExperts()
    layer0 = FakeLayerWrapper(resolved_experts, _Router())  # router + experts resolve fine
    layer1 = FakeLayerWrapper(None, _Router())  # router resolves, experts do NOT -> should raise
    layer2 = FakeLayerWrapper(None, None)  # no router at all -> legitimately dense, exempt

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [layer0, layer1, layer2]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, bias=False)

    m = _Model()

    import pytest

    with pytest.raises(ValueError, match="router"):
        list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))


def test_no_fail_closed_when_dense_layers_lack_router(monkeypatch):
    """A layer with no router at all (dense layer) must not trip the per-layer guard."""
    from olive.common.hf import wrapper as wrapper_mod

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)

    class _Router(nn.Module):
        pass

    class FakeLayerWrapper:
        def __init__(self, experts, router):
            self._experts = experts
            self._router = router

        def get_experts(self, return_name=True):
            return (self._experts, "experts") if return_name else self._experts

        def get_router(self, return_name=True):
            return (self._router, "gate") if return_name else self._router

    layer0_experts = FusedExperts()
    layer0 = FakeLayerWrapper(layer0_experts, _Router())  # MoE layer, resolves fine
    layer1 = FakeLayerWrapper(None, None)  # dense layer, no router -> exempt

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [layer0, layer1]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, bias=False)
            self.experts = layer0_experts

    m = _Model()
    # Should not raise.
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert "experts.gate_up_proj" in _names(targets)


def test_no_fail_closed_for_dense_model_with_incidental_gate_submodule(monkeypatch):
    """Regression: an incidental ``mlp.gate`` submodule on a dense model must not fail closed.

    ``get_router()`` matches purely on attribute name (default lookup key ``gate``), so
    without this guard, any dense model with an unrelated ``mlp.gate`` submodule and no MoE
    config signal would incorrectly trip the "partially supported MoE architecture"
    fail-closed refusal even when ``quantize_moe=False``.
    """
    from olive.common.hf import wrapper as wrapper_mod

    class _Gate(nn.Module):
        """An unrelated dense-MLP submodule that happens to be named ``gate``."""

    class FakeLayerWrapper:
        def __init__(self, router):
            self._router = router

        def get_experts(self, return_name=True):
            return (None, None) if return_name else None

        def get_router(self, return_name=True):
            return (self._router, "gate") if return_name else self._router

    layer0 = FakeLayerWrapper(_Gate())

    class FakeWrapper:
        def __init__(self, model):
            self.model = model

        def get_layer_wrappers(self):
            return [layer0]

    monkeypatch.setattr(wrapper_mod.ModelWrapper, "from_model", classmethod(lambda cls, m: FakeWrapper(m)))

    class _Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(8, 8, bias=False)

        def config(self):
            return None

    m = _Model()
    m.config = type("Config", (), {"model_type": "dense_model_with_gate"})()

    # Should not raise, and quantize_moe=False should behave like an ordinary dense walk.
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=False))
    assert "linear" in _names(targets)


def test_gptq_then_rtn_moe_composition_skips_already_quantized(monkeypatch):
    """Regression for `Gptq`-then-`Rtn(moe=True, embeds=True)` composition.

    Emulates GPTQ having quantized only the `nn.Linear` layers first (they become
    ``QuantTensor``-backed), then runs the RTN selection with ``moe=True`` / ``embeds=True``:
    the already-quantized Linears must be skipped (kept as their GPTQ tensors) while the MoE
    experts and embeddings are newly selected — no conflict, no double quantization.
    """
    from olive.common.quant.tensor import QuantTensor

    class FusedExperts(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_up_proj = nn.Parameter(torch.zeros(4, 8, 16), requires_grad=False)
            self.down_proj = nn.Parameter(torch.zeros(4, 16, 8), requires_grad=False)

    class _MoEModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_tokens = nn.Embedding(16, 8)
            self.q_proj = nn.Linear(8, 8, bias=False)
            self.o_proj = nn.Linear(8, 8, bias=False)
            self.experts = FusedExperts()

        def get_input_embeddings(self):
            return self.embed_tokens

    m = _MoEModel()
    _install_fake_wrapper(monkeypatch, [(m.experts, "experts")])

    # Emulate GPTQ: quantize only the nn.Linear weights (8-bit here so we can tell them apart).
    for linear in (m.q_proj, m.o_proj):
        qt = QuantTensor.from_float(linear.weight.data.clone(), bits=8, group_size=8, symmetric=True)
        linear.weight = nn.Parameter(qt, requires_grad=False)

    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))

    # RTN must only pick up what GPTQ left un-quantized: the embeddings and the MoE experts.
    assert _names(targets) == ["embed_tokens", "experts.down_proj", "experts.gate_up_proj"]

    # The GPTQ-quantized Linears are untouched (still 8-bit QuantTensor).
    assert isinstance(m.q_proj.weight.data, QuantTensor)
    assert m.q_proj.weight.data.bits == 8
    assert isinstance(m.o_proj.weight.data, QuantTensor)
    assert m.o_proj.weight.data.bits == 8


class _VisionTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.patch_embed = nn.Linear(8, 8, bias=False)
        self.blocks = nn.ModuleList([nn.Linear(8, 8, bias=False)])
        self.merger = nn.Linear(8, 8, bias=False)


class _VLModel(nn.Module):
    """Composite VL model: decoder under ``model.language_model``, tower under ``model.visual``."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.model = nn.Module()
        self.model.language_model = nn.Module()
        self.model.language_model.embed_tokens = nn.Embedding(16, 8)
        self.model.language_model.linear = nn.Linear(8, 8, bias=False)
        self.model.visual = _VisionTower()
        self.model.embed_vision = nn.Module()
        self.model.embed_vision.embedding_projection = nn.Linear(8, 8, bias=False)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_input_embeddings(self):
        return self.model.language_model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head


class _FakeConfig:
    def __init__(self, vision_config=None):
        self.vision_config = vision_config


def test_vision_tower_excluded_for_composite_vl_model():
    """Regression: a VL model's vision tower must not be a PyTorch-side quantization target.

    The vision encoder is quantized separately (int8) downstream; including it here would
    double-quantize it.
    """
    m = _VLModel(_FakeConfig(vision_config=_FakeConfig()))
    targets = list(iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True))
    assert _names(targets) == [
        "lm_head",
        "model.language_model.embed_tokens",
        "model.language_model.linear",
    ]
    assert not any(name.startswith("model.visual") for _, _, name in targets)
    assert not any(name.startswith("model.embed_vision") for _, _, name in targets)


def test_vision_named_modules_kept_when_config_has_no_vision_config():
    """Text-only models are unaffected: nothing is excluded by name alone."""
    m = _VLModel(_FakeConfig(vision_config=None))
    targets = list(iter_quant_targets(m, quantize_lm_head=False, quantize_embeds=False, quantize_moe=False))
    assert "model.visual.merger" in _names(targets)
    assert "model.embed_vision.embedding_projection" in _names(targets)


def test_quantize_vision_true_includes_vision_tower():
    """``quantize_vision=True`` opts back into quantizing the vision tower.

    Some callers quantize a VL model end-to-end in a single PyTorch-side pass, with no
    separate downstream (e.g. ONNX) vision-quantization step -- ``quantize_vision=True`` must
    let them include the vision tower instead of silently leaving it at full precision.
    """
    m = _VLModel(_FakeConfig(vision_config=_FakeConfig()))
    targets = list(
        iter_quant_targets(m, quantize_lm_head=True, quantize_embeds=True, quantize_moe=True, quantize_vision=True)
    )
    assert "model.visual.merger" in _names(targets)
    assert "model.embed_vision.embedding_projection" in _names(targets)
