# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
