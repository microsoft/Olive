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

    def test_unsupported_3d_indexing_raises_instead_of_dequantizing(self, w3d):
        """Multi-axis / advanced indexing that isn't leading-dim-only must raise, not OOM-dequant."""
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        with pytest.raises(RuntimeError, match="Unsupported indexing pattern"):
            _ = qt[:, 0, :]

    def test_unsupported_3d_op_raises_during_onnx_export(self, w3d, monkeypatch):
        """Central guard: any 3D QuantTensor op reaching the dense fallback under export must raise.

        ``torch.index_select`` is an op that is not individually special-cased, so it exercises the
        central ``_maybe_dense`` rejection rather than an op-specific check.
        """
        qt = QuantTensor.from_float(w3d, bits=4, symmetric=False, group_size=32)
        monkeypatch.setattr(torch.onnx, "is_in_onnx_export", lambda: True)
        with pytest.raises(RuntimeError, match=r"ModelBuilder|Mobius|MoE"):
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
