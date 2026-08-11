# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Unit tests for ``olive.common.quant.state_dict``.

The focus is :func:`refresh_quant_tensor_refs`'s handling of **tied/aliased** hosting
modules (``lm_head`` tied to ``model.embed_tokens`` install the *same* ``QuantTensor``
object on two modules) and its fail-closed behaviour for incomplete checkpoints.
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from olive.common.quant.state_dict import install_quant_tensor_param, refresh_quant_tensor_refs
from olive.common.quant.tensor import QuantTensor

OUT_FEATURES = 8
IN_FEATURES = 32
BITS = 4
GROUP_SIZE = 16
N_GROUPS = IN_FEATURES // GROUP_SIZE
PACKED_IN = IN_FEATURES * BITS // 8


PACKED_GROUPS = N_GROUPS * BITS // 8


def _placeholder_qt(symmetric: bool = True) -> QuantTensor:
    return QuantTensor.from_packed(
        qweight=torch.zeros(OUT_FEATURES, PACKED_IN, dtype=torch.uint8),
        scales=torch.zeros(OUT_FEATURES, N_GROUPS, dtype=torch.float32),
        qzeros=None if symmetric else torch.zeros(OUT_FEATURES, PACKED_GROUPS, dtype=torch.uint8),
        bits=BITS,
        group_size=GROUP_SIZE,
        symmetric=symmetric,
        shape=(OUT_FEATURES, IN_FEATURES),
        dtype=torch.float32,
        is_placeholder=True,
    )


def _single_quant_module(symmetric: bool = True) -> nn.Module:
    """Build a one-layer model whose ``linear1.weight`` is an unloaded quantized placeholder."""
    model = nn.Sequential()
    model.add_module("linear1", nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    install_quant_tensor_param(model.linear1, "weight", _placeholder_qt(symmetric=symmetric))
    return model


class _TiedModel(nn.Module):
    """Two modules hosting the *same* ``QuantTensor`` object, like tied word embeddings.

    ``src_first`` controls child registration order, which is exactly ``named_modules()``
    iteration order — the thing the old last-write-wins implementation was accidentally
    sensitive to.
    """

    def __init__(self, src_first: bool = True):
        super().__init__()
        src = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        dst = nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False)
        if src_first:
            self.src = src
            self.dst = dst
        else:
            self.dst = dst
            self.src = src

        install_quant_tensor_param(self.src, "weight", _placeholder_qt())
        shared = self.src._parameters["weight"]

        # Mimic ``tie_quant_word_embeddings``: alias ``dst``'s buffer dict entries to the
        # buffer *objects* ``src`` holds right now (a one-time snapshot, not a live link)
        # and share the very same parameter object.
        for name in ("weight_qweight", "weight_scales"):
            dst._non_persistent_buffers_set.add(name)
            dst._buffers[name] = self.src._buffers[name]
        dst._parameters["weight"] = shared

    @property
    def shared_param(self) -> QuantTensor:
        return self.src._parameters["weight"]


def _simulate_checkpoint_load(model: _TiedModel) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Replace ``src``'s buffer objects with freshly-loaded real data (HF ``setattr`` loader).

    ``dst``'s buffer dict keeps pointing at the *old* placeholder objects, reproducing the
    exact state the loader leaves behind for tied quantized weights.
    """
    reference = torch.randn(OUT_FEATURES, IN_FEATURES)
    real = QuantTensor.from_float(reference, bits=BITS, symmetric=True, group_size=GROUP_SIZE)
    model.src._buffers["weight_qweight"] = real.qweight
    model.src._buffers["weight_scales"] = real.scales
    return real.qweight, real.scales, real.to_dense()


@pytest.mark.parametrize("src_first", [True, False])
def test_refresh_binds_shared_quant_tensor_to_freshly_loaded_buffers(src_first: bool):
    """The shared QuantTensor must bind to the loaded buffers, whatever the module order.

    Regression: ``refresh_quant_tensor_refs`` used to rebind once per *hosting module*, so
    for tied weights the last-visited module won — and the alias module holds stale
    placeholder buffers, silently zeroing the weight.
    """
    model = _TiedModel(src_first=src_first)
    fresh_qweight, fresh_scales, expected = _simulate_checkpoint_load(model)

    refresh_quant_tensor_refs(model, checkpoint_keys={"src.weight_qweight", "src.weight_scales"})

    shared = model.shared_param
    assert shared.qweight is fresh_qweight
    assert shared.scales is fresh_scales
    assert shared.is_placeholder is False
    torch.testing.assert_close(shared.to_dense(), expected)

    # Aliased hosting module must agree at the ``_buffers`` dict level too, so
    # ``state_dict()`` / save and live forward computation cannot diverge.
    assert model.dst._buffers["weight_qweight"] is fresh_qweight
    assert model.dst._buffers["weight_scales"] is fresh_scales
    assert model.dst._parameters["weight"] is shared


@pytest.mark.parametrize("src_first", [True, False])
def test_refresh_is_order_independent_with_identity_heuristic(src_first: bool):
    """Same fix, but through the ``checkpoint_keys=None`` identity-heuristic path."""
    model = _TiedModel(src_first=src_first)
    fresh_qweight, fresh_scales, expected = _simulate_checkpoint_load(model)

    refresh_quant_tensor_refs(model, checkpoint_keys=None)

    shared = model.shared_param
    assert shared.qweight is fresh_qweight
    assert shared.scales is fresh_scales
    assert shared.is_placeholder is False
    torch.testing.assert_close(shared.to_dense(), expected)
    assert model.dst._buffers["weight_qweight"] is fresh_qweight


def test_refresh_raises_when_checkpoint_manifest_omits_a_parameter():
    """Fail closed: a known manifest missing a quantized parameter must raise, not zero it."""
    model = nn.Sequential()
    model.add_module("linear1", nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    model.add_module("linear2", nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    install_quant_tensor_param(model.linear1, "weight", _placeholder_qt())
    install_quant_tensor_param(model.linear2, "weight", _placeholder_qt())

    with pytest.raises(RuntimeError, match=r"missing weights for: linear2\.weight"):
        refresh_quant_tensor_refs(model, checkpoint_keys={"linear1.weight_qweight", "linear1.weight_scales"})


def test_refresh_does_not_raise_for_tied_alias_absent_from_manifest():
    """Only the tie *source*'s keys are persisted; the alias must not be reported missing."""
    model = _TiedModel()
    _simulate_checkpoint_load(model)

    # ``dst.weight_qweight`` is intentionally absent from the manifest (non-persistent).
    refresh_quant_tensor_refs(model, checkpoint_keys={"src.weight_qweight", "src.weight_scales"})

    assert model.shared_param.is_placeholder is False


def test_refresh_does_not_raise_when_loader_remapped_the_checkpoint_key():
    """``checkpoint_keys`` holds *raw on-disk* names, which HF may remap while loading.

    ``save_pretrained(save_original_format=True)`` writes MoE experts as legacy per-expert
    keys that the loader fuses back into ``experts.gate_up_proj_qweight``; the fused name is
    therefore absent from the raw manifest even though real data *was* loaded. Fail-closed
    must require both signals (manifest miss **and** untouched buffers) to agree.
    """
    model = nn.Sequential()
    model.add_module("linear1", nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    install_quant_tensor_param(model.linear1, "weight", _placeholder_qt())

    real = QuantTensor.from_float(
        torch.randn(OUT_FEATURES, IN_FEATURES), bits=BITS, symmetric=True, group_size=GROUP_SIZE
    )
    model.linear1._buffers["weight_qweight"] = real.qweight
    model.linear1._buffers["weight_scales"] = real.scales

    refresh_quant_tensor_refs(model, checkpoint_keys={"some.legacy.key_qweight"})

    shared = model.linear1._parameters["weight"]
    assert shared.qweight is real.qweight
    assert shared.is_placeholder is False


def test_refresh_with_unknown_checkpoint_keys_stays_permissive():
    """Regression: ``checkpoint_keys=None`` keeps the permissive identity heuristic.

    Nothing was loaded, so the parameter stays a placeholder — but we must NOT raise,
    because the checkpoint format/manifest is unknown and "not loaded" cannot be
    distinguished from "loaded in place".
    """
    model = nn.Sequential()
    model.add_module("linear1", nn.Linear(IN_FEATURES, OUT_FEATURES, bias=False))
    install_quant_tensor_param(model.linear1, "weight", _placeholder_qt())

    refresh_quant_tensor_refs(model, checkpoint_keys=None)

    assert model.linear1._parameters["weight"].is_placeholder is True


def test_refresh_ignores_modules_without_quant_tensors():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())
    refresh_quant_tensor_refs(model, checkpoint_keys=set())


# ----------------------------------------------------------------------
# Partial loads must not count as "loaded" (fail-closed false-negative gap).
# ----------------------------------------------------------------------


def test_refresh_raises_when_manifest_has_scales_but_not_qweight():
    """A manifest carrying only *some* of a parameter's buffers is an incomplete checkpoint.

    Regression: the fail-closed check used to look at ``<pname>_qweight`` membership alone
    (and, worse, treated *any* single signal as proof of a load), so a truncated checkpoint
    that contains ``weight_scales`` but not ``weight_qweight`` was reported as loaded — the
    placeholder flag was cleared over an all-zero ``qweight``.
    """
    model = _single_quant_module()
    qt = model.linear1._parameters["weight"]

    # The loader wrote the one key it found; ``weight_qweight`` stays the zero placeholder.
    model.linear1._buffers["weight_scales"] = torch.randn(OUT_FEATURES, N_GROUPS)

    with pytest.raises(RuntimeError, match=r"missing weights for: linear1\.weight"):
        refresh_quant_tensor_refs(model, checkpoint_keys={"linear1.weight_scales"})

    assert qt.is_placeholder is True
    assert int(qt.qweight.sum()) == 0


def test_refresh_raises_when_manifest_omits_qzeros_for_asymmetric_parameter():
    """``_qzeros`` is mandatory when the QuantTensor is asymmetric."""
    model = _single_quant_module(symmetric=False)
    qt = model.linear1._parameters["weight"]

    with pytest.raises(RuntimeError, match=r"missing weights for: linear1\.weight"):
        refresh_quant_tensor_refs(model, checkpoint_keys={"linear1.weight_qweight", "linear1.weight_scales"})

    assert qt.is_placeholder is True


def test_refresh_treats_symmetric_parameter_without_qzeros_as_fully_loaded():
    """Symmetric quantization has no ``_qzeros`` buffer — don't demand a key for it."""
    model = _single_quant_module()

    refresh_quant_tensor_refs(model, checkpoint_keys={"linear1.weight_qweight", "linear1.weight_scales"})

    assert model.linear1._parameters["weight"].is_placeholder is False


def test_refresh_treats_asymmetric_parameter_with_all_keys_as_fully_loaded():
    """The asymmetric counterpart: all three keys present means fully loaded."""
    model = _single_quant_module(symmetric=False)

    refresh_quant_tensor_refs(
        model,
        checkpoint_keys={"linear1.weight_qweight", "linear1.weight_scales", "linear1.weight_qzeros"},
    )

    assert model.linear1._parameters["weight"].is_placeholder is False


def test_refresh_identity_heuristic_ignores_partial_buffer_swap():
    """``checkpoint_keys=None``: a partial buffer swap must not clear ``is_placeholder``.

    The identity heuristic used to OR across ``qweight``/``scales``/``qzeros``, so swapping
    any single buffer object marked the whole parameter loaded. It now requires *all*
    mandatory buffers to have been replaced. The permissive contract of this code path is
    unchanged (nothing is raised when the manifest is unknown) — the parameter simply stays
    a placeholder.
    """
    model = _single_quant_module()
    qt = model.linear1._parameters["weight"]
    placeholder_qweight = qt.qweight

    model.linear1._buffers["weight_scales"] = torch.randn(OUT_FEATURES, N_GROUPS)

    refresh_quant_tensor_refs(model, checkpoint_keys=None)

    assert qt.is_placeholder is True
    assert qt.qweight is placeholder_qweight
    assert int(qt.qweight.sum()) == 0


def test_refresh_identity_heuristic_ignores_missing_qzeros_swap():
    """Asymmetric variant: ``qweight``/``scales`` swapped but ``qzeros`` left stale."""
    model = _single_quant_module(symmetric=False)
    qt = model.linear1._parameters["weight"]
    placeholder_qzeros = qt.qzeros

    model.linear1._buffers["weight_qweight"] = torch.randint(0, 255, (OUT_FEATURES, PACKED_IN), dtype=torch.uint8)
    model.linear1._buffers["weight_scales"] = torch.randn(OUT_FEATURES, N_GROUPS)

    refresh_quant_tensor_refs(model, checkpoint_keys=None)

    assert qt.is_placeholder is True
    assert qt.qzeros is placeholder_qzeros


def test_refresh_identity_heuristic_accepts_full_buffer_swap():
    """Sanity check that the AND-ed identity heuristic still recognises a real full load."""
    model = _single_quant_module(symmetric=False)
    qt = model.linear1._parameters["weight"]

    model.linear1._buffers["weight_qweight"] = torch.randint(0, 255, (OUT_FEATURES, PACKED_IN), dtype=torch.uint8)
    model.linear1._buffers["weight_scales"] = torch.randn(OUT_FEATURES, N_GROUPS)
    model.linear1._buffers["weight_qzeros"] = torch.randint(0, 255, (OUT_FEATURES, PACKED_GROUPS), dtype=torch.uint8)

    refresh_quant_tensor_refs(model, checkpoint_keys=None)

    assert qt.is_placeholder is False
    assert qt.qweight is model.linear1._buffers["weight_qweight"]
    assert qt.qzeros is model.linear1._buffers["weight_qzeros"]
