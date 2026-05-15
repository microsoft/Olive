# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
