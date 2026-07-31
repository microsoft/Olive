# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=redefined-outer-name,not-callable
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from olive.common.quant.tensor import QuantTensor


@pytest.fixture
def w2d():
    torch.manual_seed(0)
    return torch.randn(64, 128, dtype=torch.float32)


@pytest.fixture
def w3d():
    torch.manual_seed(0)
    return torch.randn(4, 32, 128, dtype=torch.float32)


class TestQuantTensor2D:
    def test_shape_dtype_device_preserved(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        assert qt.shape == w2d.shape
        assert qt.dtype == w2d.dtype
        assert qt.device == w2d.device
        assert qt.requires_grad is False

    def test_inner_buffer_layout(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        # 4-bit packed → in_features / 2
        assert qt.qweight.shape == (64, 64)
        assert qt.qweight.dtype == torch.uint8
        # groupwise scales: (out, num_groups)
        assert qt.scales.shape == (64, 128 // 32)
        # symmetric → no zero_points
        assert qt.qzeros is None

    def test_asymmetric_has_qzeros(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=False, group_size=32)
        assert qt.qzeros is not None
        assert qt.qzeros.dtype == torch.uint8

    def test_to_dense_round_trip(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=False, group_size=32)
        dense = qt.to_dense()
        assert dense.shape == w2d.shape
        # Round trip should be close (4-bit groupwise is reasonably accurate)
        assert (dense - w2d).abs().mean().item() < 0.1

    def test_dispatches_through_f_linear(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        x = torch.randn(2, 128)
        out_quant = F.linear(x, qt)
        out_dense = F.linear(x, qt.to_dense())
        assert torch.allclose(out_quant, out_dense, atol=1e-5)

    def test_nn_parameter_preserves_subclass(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        p = nn.Parameter(qt, requires_grad=False)
        assert isinstance(p, QuantTensor)
        assert isinstance(p.data, QuantTensor)

    def test_nn_linear_forward_with_quant_tensor_weight(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        layer = nn.Linear(128, 64, bias=False)
        layer.weight = nn.Parameter(qt, requires_grad=False)
        x = torch.randn(2, 128)
        out_layer = layer(x)
        out_ref = F.linear(x, qt.to_dense())
        assert torch.allclose(out_layer, out_ref, atol=1e-5)

    def test_model_to_dtype_propagates(self, w2d):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=False, group_size=32)

        class M(nn.Module):
            def __init__(self):
                super().__init__()
                self.lin = nn.Linear(128, 64, bias=False)

        m = M()
        m.lin.weight = nn.Parameter(qt, requires_grad=False)
        m = m.to(torch.float16)
        # dtype follows the wrapper subclass; scales are floating-point
        assert m.lin.weight.dtype == torch.float16
        assert m.lin.weight.scales.dtype == torch.float16
        # qweight is uint8 — non-floating-point, kept as-is
        assert m.lin.weight.qweight.dtype == torch.uint8

    def test_nn_embedding_forward(self, w2d):
        # 64 embeddings of dim 128
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        emb = nn.Embedding(64, 128)
        emb.weight = nn.Parameter(qt, requires_grad=False)
        ids = torch.tensor([0, 5, 60])
        out = emb(ids)
        out_ref = F.embedding(ids, qt.to_dense())
        assert torch.allclose(out, out_ref, atol=1e-5)


class TestQuantTensor3D:
    def test_3d_shape(self, w3d):
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        assert qt.shape == w3d.shape
        assert qt.qweight.shape == (4, 32, 64)
        assert qt.scales.shape == (4, 32, 4)
        assert qt.qzeros is not None
        assert qt.qzeros.shape == (4, 32, 2)

    def test_3d_round_trip(self, w3d):
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        dense = qt.to_dense()
        assert (dense - w3d).abs().mean().item() < 0.1

    def test_slice_returns_2d_quant_tensor(self, w3d):
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        sliced = qt[2]
        assert isinstance(sliced, QuantTensor)
        assert sliced.shape == w3d[2].shape
        # F.linear over the slice
        x = torch.randn(2, 128)
        out = F.linear(x, sliced)
        out_ref = F.linear(x, qt.to_dense()[2])
        assert torch.allclose(out, out_ref, atol=1e-5)


class TestQuantTensorOnnxExportGuards:
    def test_linear_raises_when_in_onnx_export(self, w2d, monkeypatch):
        qt = QuantTensor.from_float(w2d, bits=4, symmetric=True, group_size=32)
        x = torch.randn(1, 128)
        # Simulate being inside ONNX export
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        with pytest.raises(RuntimeError, match="QuantTensor cannot be traced"):
            F.linear(x, qt)


class TestQuantTensor3DExpertRouting:
    def test_tensor_index_routing_preserves_quantized_storage(self, w3d):
        """Advanced/tensor-index expert selection (how real MoE routes) stays quantized."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        expert_ids = torch.tensor([0, 2, 2, 1])
        selected = qt[expert_ids]
        assert isinstance(selected, QuantTensor)
        assert selected.shape == (4, *w3d.shape[1:])
        # Values match a dense gather.
        ref = qt.to_dense()[expert_ids]
        assert torch.allclose(selected.to_dense(), ref, atol=1e-6)

    def test_list_index_routing_preserves_quantized_storage(self, w3d):
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        selected = qt[[0, 3]]
        assert isinstance(selected, QuantTensor)
        assert selected.shape == (2, *w3d.shape[1:])

    def test_tuple_leading_only_index_preserves_quantized_storage(self, w3d):
        """M2: tuple-form leading-dim indexing ``w[expert_ids, :, :]`` must stay quantized."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        expert_ids = torch.tensor([0, 2])
        selected = qt[expert_ids, :, :]
        assert isinstance(selected, QuantTensor)
        assert selected.shape == (2, *w3d.shape[1:])
        ref = qt.to_dense()[expert_ids, :, :]
        assert torch.allclose(selected.to_dense(), ref, atol=1e-6)

    def test_rank2_tensor_index_preserves_quantized_storage(self, w3d):
        """M2: rank>1 leading tensor indices (e.g. (tokens, k) top-k routing) stay quantized.

        The resulting QuantTensor may end up rank > 3 (here 4D) -- that's fine; it's a cheap,
        storage-only view. Only a later dense op on it raises (see the >3D test below).
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        idx = torch.tensor([[0, 1], [2, 3]])
        selected = qt[idx]
        assert isinstance(selected, QuantTensor)
        assert selected.shape == (2, 2, *w3d.shape[1:])
        assert selected.dim() == 4

    def test_gt_3d_quant_tensor_to_dense_raises(self, w3d):
        """A >3D QuantTensor (constructed via a rank>1 leading index) refuses to_dense()."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        idx = torch.tensor([[0, 1], [2, 3]])
        selected = qt[idx]
        with pytest.raises(NotImplementedError, match="2D / 3D"):
            selected.to_dense()

    def test_unsupported_3d_indexing_raises_instead_of_dequantizing(self, w3d):
        """Multi-axis / advanced indexing that isn't leading-dim-only must raise, not OOM-dequant."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        with pytest.raises(RuntimeError, match="Unsupported indexing pattern"):
            _ = qt[:, 0, :]

    def test_rank2_bool_mask_raises_instead_of_misclassifying(self, w3d):
        """Round-2 regression: a rank>1 boolean mask must raise, not silently misclassify shape.

        Previously any boolean tensor was treated as "leading-dim only", producing a
        QuantTensor whose ``.shape`` metadata disagreed with ``.to_dense()``'s actual shape.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        mask = torch.ones((4, 8), dtype=torch.bool)
        with pytest.raises(RuntimeError, match="Unsupported indexing pattern"):
            _ = qt[mask]

    def test_1d_bool_mask_preserves_quantized_storage_and_shape(self, w3d):
        """A 1-D boolean mask over the leading (expert) dim is a safe, quantized-preserving index."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        mask = torch.tensor([True, False, True, False])
        selected = qt[mask]
        assert isinstance(selected, QuantTensor)
        assert selected.shape == (2, *w3d.shape[1:])
        assert selected.to_dense().shape == selected.shape
        ref = qt.to_dense()[mask]
        assert torch.allclose(selected.to_dense(), ref, atol=1e-6)

    @pytest.mark.parametrize(
        "idx",
        [
            0,
            slice(0, 2),
            [0, 2, 3],
            torch.tensor([0, 2]),
            torch.tensor([[0, 1], [2, 3]]),
            torch.tensor([True, False, True, False]),
        ],
    )
    def test_getitem_shape_matches_dense_for_every_accepted_index_form(self, w3d, idx):
        """Property test: for every accepted index form, ``.shape`` metadata must agree with dense.

        This is the invariant both this bug and any future ``_getitem`` extension must preserve.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        result = qt[idx]
        dense_result = qt.to_dense()[idx]
        assert tuple(result.shape) == tuple(dense_result.shape)
        if isinstance(result, QuantTensor) and result.dim() in (2, 3):
            assert torch.allclose(result.to_dense(), dense_result, atol=1e-6)

    def test_unsupported_3d_op_raises_during_onnx_export(self, w3d, monkeypatch):
        """Central guard: any 3D QuantTensor op reaching the dense fallback under export must raise.

        ``torch.index_select`` is an op that is not individually special-cased, so it exercises the
        central ``_maybe_dense`` rejection rather than an op-specific check.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        with pytest.raises(RuntimeError, match=r"ModelBuilder|Mobius|MoE"):
            torch.index_select(qt, 1, torch.tensor([0, 1]))

    def test_unsupported_3d_op_raises_in_eager_mode(self, w3d):
        """M1: the same central guard must also raise in plain eager mode (no ONNX export).

        Previously an unregistered op on a 3D QuantTensor fell through to the generic
        ``__torch_dispatch__`` fallback and fully dequantized the (potentially huge) expert
        tensor in eager mode -- an OOM risk this guard must refuse instead.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        with pytest.raises(RuntimeError, match=r"OOM|dequant|refused"):
            torch.index_select(qt, 1, torch.tensor([0, 1]))

    @pytest.mark.parametrize(
        "op",
        [
            lambda qt: qt.transpose(-2, -1),
            lambda qt: qt.reshape(4, -1),
            lambda qt: qt.view(4, -1),
            lambda qt: qt.permute(0, 2, 1),
            lambda qt: qt.flatten(),
            lambda qt: torch.transpose(qt, -2, -1),
        ],
    )
    def test_movement_ops_raise_instead_of_silently_misbehaving(self, w3d, op):
        """Shape-movement / view ops must raise a clear error rather than produce a malformed tensor."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        with pytest.raises(RuntimeError, match="storage-only"):
            op(qt)

    def test_movement_op_under_onnx_export_raises_moe_message(self, w3d, monkeypatch):
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        with pytest.raises(RuntimeError, match=r"ModelBuilder|Mobius|MoE"):
            qt.transpose(-2, -1)


class TestQuantTensorPlaceholderInit:
    def test_inplace_initializers_are_noop_on_placeholder(self, w3d):
        """Regression: HF's ``_initialize_weights`` calls in-place initializers on placeholders.

        HF's ``PreTrainedModel._initialize_weights`` calls in-place initializers (e.g.
        ``nn.init.normal_``) on freshly-installed placeholder QuantTensor params before the
        checkpoint's real buffers are loaded. These must be safe (harmless) no-ops rather
        than raising (the M1 eager-3D guard would otherwise break real HF model loading for
        MoE / fused-3D targets) or silently dequantizing just to throw the result away.
        """
        qt = QuantTensor.from_packed(
            qweight=torch.zeros(4, 32, 64, dtype=torch.uint8),
            scales=torch.zeros(4, 32, 4, dtype=torch.float32),
            qzeros=torch.zeros(4, 32, 2, dtype=torch.uint8),
            bits=4,
            group_size=32,
            symmetric=False,
            shape=(4, 32, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        assert qt.is_placeholder is True
        # Must not raise.
        torch.nn.init.normal_(qt, mean=0.0, std=0.02)
        torch.nn.init.zeros_(qt)
        assert isinstance(qt, QuantTensor)

    def test_inplace_initializers_raise_on_real_quant_tensor(self, w3d):
        """Round-2 regression: in-place init on a *real* (non-placeholder) QuantTensor must raise.

        Silently no-oping here would let ``torch.nn.init.zeros_(real_qt)`` "succeed" while the
        dequantized values stay unchanged -- a silent data-integrity bug.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        assert qt.is_placeholder is False
        with pytest.raises(RuntimeError, match="In-place initializer"):
            torch.nn.init.zeros_(qt)
        with pytest.raises(RuntimeError, match="In-place initializer"):
            torch.nn.init.normal_(qt, mean=0.0, std=0.02)

    def test_is_placeholder_survives_nn_parameter_detach_and_to(self, w3d):
        """The placeholder flag must survive ``nn.Parameter(qt)``'s ``.detach()`` and ``.to(...)``.

        ``torch.nn.Parameter(qt, requires_grad=False)`` for a tensor subclass returns
        ``qt.detach()``, which constructs a *new* QuantTensor via ``_apply_fn_to_data``. If the
        flag were lost here, HF's ``_initialize_weights`` would raise instead of no-oping on the
        placeholder it installs before checkpoint loading.
        """
        qt = QuantTensor.from_packed(
            qweight=torch.zeros(4, 32, 64, dtype=torch.uint8),
            scales=torch.zeros(4, 32, 4, dtype=torch.float32),
            qzeros=torch.zeros(4, 32, 2, dtype=torch.uint8),
            bits=4,
            group_size=32,
            symmetric=False,
            shape=(4, 32, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        p = torch.nn.Parameter(qt, requires_grad=False)
        assert isinstance(p, QuantTensor)
        assert p.is_placeholder is True
        p2 = p.to(torch.float16)
        assert isinstance(p2, QuantTensor)
        assert p2.is_placeholder is True
        assert p[0].is_placeholder is True

    def test_refresh_quant_tensor_refs_clears_placeholder_flag(self):
        """``refresh_quant_tensor_refs`` binds real checkpoint buffers and clears the flag."""
        from olive.common.quant.state_dict import refresh_quant_tensor_refs

        qt = QuantTensor.from_packed(
            qweight=torch.zeros(64, 64, dtype=torch.uint8),
            scales=torch.zeros(64, 4, dtype=torch.float32),
            qzeros=None,
            bits=4,
            group_size=32,
            symmetric=True,
            shape=(64, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        layer = nn.Linear(128, 64, bias=False)
        layer.weight = nn.Parameter(qt, requires_grad=False)
        assert layer.weight.is_placeholder is True

        # Simulate HF's checkpoint buffer load: real ``<pname>_qweight`` / ``_scales`` buffers
        # land in ``_buffers`` under the naming convention ``buffer_names`` expects.
        layer.register_buffer("weight_qweight", torch.randint(0, 255, (64, 64), dtype=torch.uint8))
        layer.register_buffer("weight_scales", torch.randn(64, 4, dtype=torch.float32))

        refresh_quant_tensor_refs(layer)
        assert layer.weight.is_placeholder is False

    def test_refresh_quant_tensor_refs_keeps_placeholder_when_key_missing(self):
        """Round-3 regression (#2598 item 3): a missing checkpoint key must not clear the flag.

        ``refresh_quant_tensor_refs`` is called once for the *whole model* after HF's loader
        finishes, with no per-parameter "was this key actually in the checkpoint" signal. A
        parameter whose key was missing from the checkpoint keeps the exact placeholder buffer
        objects installed by ``install_quant_tensor_param`` -- only a real
        ``load_state_dict(..., assign=True)`` replaces the buffer objects. Unconditionally
        clearing ``is_placeholder`` here would make a later in-place initializer (which HF may
        still call for missing-key parameters) raise instead of safely no-oping.
        """
        from olive.common.quant.state_dict import refresh_quant_tensor_refs

        qt = QuantTensor.from_packed(
            qweight=torch.zeros(64, 64, dtype=torch.uint8),
            scales=torch.zeros(64, 4, dtype=torch.float32),
            qzeros=None,
            bits=4,
            group_size=32,
            symmetric=True,
            shape=(64, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        layer = nn.Linear(128, 64, bias=False)
        # ``nn.Parameter(qt)`` for a tensor subclass returns ``qt.detach()``, which produces
        # *new* (storage-aliased) inner tensor objects -- so read them back off
        # ``layer.weight`` (like ``install_quant_tensor_param`` does), not off ``qt`` itself.
        layer.weight = nn.Parameter(qt, requires_grad=False)
        # Register the placeholder buffers aliasing the exact same tensor objects the
        # installed QuantTensor parameter already holds -- i.e. simulate "checkpoint load
        # ran, but this parameter's key was missing so its buffers were never reassigned".
        layer.register_buffer("weight_qweight", layer.weight.qweight)
        layer.register_buffer("weight_scales", layer.weight.scales)

        refresh_quant_tensor_refs(layer)
        assert layer.weight.is_placeholder is True
        # Must still be a safe no-op, not a raise.
        torch.nn.init.zeros_(layer.weight)

    def test_copy_into_placeholder_clears_flag(self, w3d):
        """Round-3 regression (#2598 item 2): ``copy_`` must propagate ``is_placeholder``.

        Copying real (non-placeholder) data into a placeholder ``QuantTensor`` makes it real
        too. If ``is_placeholder`` were left ``True``, a later in-place initializer (e.g. a
        module re-init call) could still silently no-op and discard the just-copied real data.
        """
        placeholder = QuantTensor.from_packed(
            qweight=torch.zeros(4, 32, 64, dtype=torch.uint8),
            scales=torch.zeros(4, 32, 4, dtype=torch.float32),
            qzeros=torch.zeros(4, 32, 2, dtype=torch.uint8),
            bits=4,
            group_size=32,
            symmetric=False,
            shape=(4, 32, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        real = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        assert placeholder.is_placeholder is True
        assert real.is_placeholder is False

        placeholder.copy_(real)
        assert placeholder.is_placeholder is False
        # Data integrity: an in-place initializer must now raise (real data present),
        # not silently no-op.
        with pytest.raises(RuntimeError, match="In-place initializer"):
            torch.nn.init.zeros_(placeholder)

    def test_copy_from_placeholder_keeps_flag(self, w3d):
        """Copying a placeholder's (throwaway) data into another QuantTensor keeps it a placeholder."""
        src_placeholder = QuantTensor.from_packed(
            qweight=torch.zeros(4, 32, 64, dtype=torch.uint8),
            scales=torch.zeros(4, 32, 4, dtype=torch.float32),
            qzeros=torch.zeros(4, 32, 2, dtype=torch.uint8),
            bits=4,
            group_size=32,
            symmetric=False,
            shape=(4, 32, 128),
            dtype=torch.float32,
            is_placeholder=True,
        )
        dst_real = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        assert dst_real.is_placeholder is False

        dst_real.copy_(src_placeholder)
        assert dst_real.is_placeholder is True
        # Must not raise now that it's (again) a placeholder.
        torch.nn.init.zeros_(dst_real)
