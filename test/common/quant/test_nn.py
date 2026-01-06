# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math

import pytest
import torch
import torch.nn as nn

from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.common.quant.utils import WeightQuantizer


class TestQuantModule:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("group_size", [-1, 16, 32])
    def test_initialization(self, bits, symmetric, group_size):
        """Test QuantModule initialization with various parameters."""
        rows, cols = 64, 128

        # Since QuantModule is abstract, we'll use QuantLinear
        qmodule = QuantLinear(
            in_features=cols,
            out_features=rows,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
        )

        assert qmodule.rows == rows
        assert qmodule.cols == cols
        assert qmodule.quantizer.bits == bits
        assert qmodule.quantizer.symmetric == symmetric
        assert qmodule.quantizer.group_size == group_size

    def test_invalid_bits(self):
        """Test that invalid bits raise ValueError."""
        with pytest.raises(ValueError, match="Only 2-bit, 4-bit and 8-bit quantization supported"):
            QuantLinear(10, 20, bits=16, symmetric=True, group_size=-1)

    def test_invalid_group_size(self):
        """Test that invalid group size raises ValueError."""
        with pytest.raises(ValueError, match="group_size must be >= 16 and power of 2"):
            QuantLinear(10, 20, bits=4, symmetric=True, group_size=15)

        with pytest.raises(ValueError, match="group_size must be >= 16 and power of 2"):
            QuantLinear(10, 20, bits=4, symmetric=True, group_size=24)

    def test_invalid_in_features_for_group_size(self):
        """Test that in_features must be divisible by group_size."""
        # in_features=100 is not divisible by group_size=32
        with pytest.raises(ValueError, match="cols .* must be divisible by group_size"):
            QuantLinear(in_features=100, out_features=20, bits=4, symmetric=True, group_size=32)

        # in_features=50 is not divisible by group_size=16
        with pytest.raises(ValueError, match="cols .* must be divisible by group_size"):
            QuantLinear(in_features=50, out_features=20, bits=4, symmetric=True, group_size=16)

        # This should work: in_features=64 is divisible by group_size=32
        qlinear = QuantLinear(in_features=64, out_features=20, bits=4, symmetric=True, group_size=32)
        assert qlinear.cols == 64

    def test_buffer_shapes(self):
        """Test that buffers have correct shapes."""
        rows, cols = 64, 128
        bits = 4
        packing_factor = 8 // bits

        qmodule = QuantLinear(
            in_features=cols,
            out_features=rows,
            bits=bits,
            symmetric=True,
            group_size=32,
        )

        # Check qweight shape
        assert qmodule.qweight.shape == (rows, math.ceil(cols / packing_factor))

        # Check scales shape
        quantizer = WeightQuantizer(bits=bits, symmetric=True, group_size=32, signed=False)
        expected_scale_shape = quantizer.get_qparam_shape((rows, cols))
        assert qmodule.scales.shape == expected_scale_shape

    def test_symmetric_no_qzeros(self):
        """Test that symmetric quantization has no qzeros."""
        qmodule = QuantLinear(16, 20, bits=4, symmetric=True, group_size=16)
        assert qmodule.qzeros is None

    def test_asymmetric_has_qzeros(self):
        """Test that asymmetric quantization has qzeros."""
        qmodule = QuantLinear(16, 20, bits=4, symmetric=False, group_size=16)
        assert qmodule.qzeros is not None

        packing_factor = 8 // 4
        quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=16, signed=False)
        scale_shape = quantizer.get_qparam_shape((20, 16))
        expected_qzeros_shape = (scale_shape[0], math.ceil(scale_shape[1] / packing_factor))
        assert qmodule.qzeros.shape == expected_qzeros_shape


class TestQuantLinear:
    def test_basic_initialization(self):
        """Test basic QuantLinear initialization."""
        qlinear = QuantLinear(in_features=128, out_features=256, bits=4, symmetric=True, group_size=32)
        assert qlinear.cols == 128
        assert qlinear.rows == 256
        assert qlinear.bias is not None

    def test_initialization_without_bias(self):
        """Test QuantLinear initialization without bias."""
        qlinear = QuantLinear(
            in_features=128,
            out_features=256,
            bits=4,
            symmetric=True,
            group_size=32,
            bias=False,
        )
        assert qlinear.bias is None

    def test_from_module_basic(self):
        """Test creating QuantLinear from nn.Linear."""
        linear = nn.Linear(128, 256)
        linear.weight.data.normal_(0, 0.02)

        qlinear = QuantLinear.from_module(linear, bits=4, symmetric=True, group_size=32)

        assert qlinear.cols == 128
        assert qlinear.rows == 256
        assert qlinear.bias is not None

        # Check that weights are quantized
        assert qlinear.qweight.dtype == torch.uint8
        assert qlinear.scales.dtype == linear.weight.dtype

    def test_from_module_without_bias(self):
        """Test creating QuantLinear from nn.Linear without bias."""
        linear = nn.Linear(128, 256, bias=False)
        linear.weight.data.normal_(0, 0.02)

        qlinear = QuantLinear.from_module(linear, bits=4, symmetric=True, group_size=32)

        assert qlinear.bias is None

    def test_from_module_preserves_bias(self):
        """Test that bias is preserved when converting."""
        linear = nn.Linear(128, 256)
        linear.weight.data.normal_(0, 0.02)
        linear.bias.data.fill_(0.5)

        qlinear = QuantLinear.from_module(linear, bits=4, symmetric=True, group_size=32)

        assert torch.all(qlinear.bias == 0.5)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_from_module_quantization_accuracy(self, bits, symmetric):
        """Test that quantization/dequantization is reasonably accurate."""
        linear = nn.Linear(64, 32, bias=False)
        linear.weight.data.normal_(0, 0.02)

        qlinear = QuantLinear.from_module(linear, bits=bits, symmetric=symmetric, group_size=16)

        # Dequantize and compare
        dequantized = qlinear.unpack_and_dequantize(qlinear.qweight, qlinear.scales, qlinear.qzeros)

        # Check shapes match
        assert dequantized.shape == linear.weight.shape

        # Check that dequantized weights are close to original (within quantization error)
        # The tolerance depends on the bit width and data distribution
        max_diff = torch.max(torch.abs(dequantized - linear.weight))
        # For 4-bit, we expect larger errors than 8-bit
        tolerance = 0.1 if bits == 4 else 0.05
        assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        linear = nn.Linear(128, 256)
        linear.weight.data.normal_(0, 0.02)

        qlinear = QuantLinear.from_module(linear, bits=4, symmetric=True, group_size=32)

        x = torch.randn(16, 128)
        output = qlinear(x)

        assert output.shape == (16, 256)

    def test_forward_batch_shape(self):
        """Test forward pass with batched input."""
        linear = nn.Linear(64, 128)
        linear.weight.data.normal_(0, 0.02)

        qlinear = QuantLinear.from_module(linear, bits=4, symmetric=True, group_size=16)

        x = torch.randn(8, 32, 64)
        output = qlinear(x)

        assert output.shape == (8, 32, 128)

    def test_forward_invalid_shape(self):
        """Test that forward raises error for invalid input shape."""
        qlinear = QuantLinear(in_features=128, out_features=256, bits=4, symmetric=True, group_size=32)

        x = torch.randn(16, 64)  # Wrong input features

        with pytest.raises(AssertionError, match="Input shape .* does not match in_features"):
            qlinear(x)

    def test_from_tensors_with_precomputed_params(self):
        """Test creating QuantLinear with precomputed scales and zero points."""
        weight = torch.randn(256, 128)

        # Compute quantization parameters
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32, signed=False)
        scales, zero_points = quantizer.find_qparams(weight)

        # Create QuantLinear with precomputed params
        qlinear = QuantLinear.from_tensors(
            in_features=128,
            out_features=256,
            weight=weight,
            bits=4,
            symmetric=True,
            group_size=32,
            scales=scales,
            zero_points=zero_points,
            bias=False,
        )

        assert qlinear.scales.shape == scales.shape
        assert torch.all(qlinear.scales == scales)

    def test_from_tensors_quantized_weights(self):
        """Test creating QuantLinear from already quantized weights."""
        weight = torch.randn(256, 128)

        # Quantize weights
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32, signed=False)
        scales, zero_points = quantizer.find_qparams(weight)
        qweight = quantizer.quantize(weight, scales, zero_points)

        # Create QuantLinear from quantized weights
        qlinear = QuantLinear.from_tensors(
            in_features=128,
            out_features=256,
            weight=qweight,
            bits=4,
            symmetric=True,
            group_size=32,
            scales=scales,
            zero_points=zero_points,
            quantized=True,
            bias=False,
        )

        assert qlinear.cols == 128
        assert qlinear.rows == 256

    def test_from_tensors_missing_params_for_quantized(self):
        """Test that error is raised when params are missing for quantized weights."""
        qweight = torch.randint(0, 16, (256, 128))

        with pytest.raises(ValueError, match="scales and/or zero_points missing"):
            QuantLinear.from_tensors(
                in_features=128,
                out_features=256,
                weight=qweight,
                bits=4,
                symmetric=True,
                group_size=32,
                quantized=True,
                bias=False,
            )

    def test_from_tensors_invalid_symmetric_zero_points(self):
        """Test that error is raised for invalid zero points in symmetric quantization."""
        weight = torch.randn(256, 128)

        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32, signed=False)
        scales, zero_points = quantizer.find_qparams(weight)

        # Modify zero points to be invalid for symmetric quantization
        zero_points.fill_(0)

        with pytest.raises(ValueError, match="Zero points must be equal to midq for symmetric quantization"):
            QuantLinear.from_tensors(
                in_features=128,
                out_features=256,
                weight=weight,
                bits=4,
                symmetric=True,
                group_size=32,
                scales=scales,
                zero_points=zero_points,
                bias=False,
            )

    def test_extra_repr(self):
        """Test string representation of QuantLinear."""
        qlinear = QuantLinear(
            in_features=128,
            out_features=256,
            bits=4,
            symmetric=True,
            group_size=32,
            bias=True,
        )

        repr_str = qlinear.extra_repr()
        assert "in_features=128" in repr_str
        assert "out_features=256" in repr_str
        assert "bits=4" in repr_str
        assert "symmetric=True" in repr_str
        assert "group_size=32" in repr_str
        assert "bias=True" in repr_str


class TestQuantEmbedding:
    def test_basic_initialization(self):
        """Test basic QuantEmbedding initialization."""
        qembed = QuantEmbedding(
            num_embeddings=1000,
            embedding_dim=256,
            bits=4,
            symmetric=True,
            group_size=32,
        )
        assert qembed.rows == 1000
        assert qembed.cols == 256
        assert qembed.padding_idx is None

    def test_initialization_with_padding_idx(self):
        """Test QuantEmbedding initialization with padding_idx."""
        qembed = QuantEmbedding(
            num_embeddings=1000,
            embedding_dim=256,
            bits=4,
            symmetric=True,
            group_size=32,
            padding_idx=0,
        )
        assert qembed.padding_idx == 0

    def test_from_module_basic(self):
        """Test creating QuantEmbedding from nn.Embedding."""
        embedding = nn.Embedding(1000, 256)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=4, symmetric=True, group_size=32)

        assert qembed.rows == 1000
        assert qembed.cols == 256
        assert qembed.padding_idx is None

    def test_from_module_with_padding_idx(self):
        """Test creating QuantEmbedding with padding_idx."""
        embedding = nn.Embedding(1000, 256, padding_idx=0)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=4, symmetric=True, group_size=32)

        assert qembed.padding_idx == 0

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    def test_from_module_quantization_accuracy(self, bits, symmetric):
        """Test that quantization/dequantization is reasonably accurate."""
        embedding = nn.Embedding(100, 64)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=bits, symmetric=symmetric, group_size=16)

        # Dequantize and compare
        dequantized = qembed.unpack_and_dequantize(qembed.qweight, qembed.scales, qembed.qzeros)

        # Check shapes match
        assert dequantized.shape == embedding.weight.shape

        # Check that dequantized weights are close to original
        max_diff = torch.max(torch.abs(dequantized - embedding.weight))
        tolerance = 0.1 if bits == 4 else 0.05
        assert max_diff < tolerance, f"Max difference {max_diff} exceeds tolerance {tolerance}"

    def test_forward_shape(self):
        """Test that forward pass produces correct output shape."""
        embedding = nn.Embedding(1000, 256)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=4, symmetric=True, group_size=32)

        x = torch.randint(0, 1000, (16,))
        output = qembed(x)

        assert output.shape == (16, 256)

    def test_forward_2d_input(self):
        """Test forward pass with 2D input."""
        embedding = nn.Embedding(1000, 256)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=4, symmetric=True, group_size=32)

        x = torch.randint(0, 1000, (8, 32))
        output = qembed(x)

        assert output.shape == (8, 32, 256)

    def test_forward_3d_input(self):
        """Test forward pass with 3D input."""
        embedding = nn.Embedding(1000, 256)
        embedding.weight.data.normal_(0, 0.02)

        qembed = QuantEmbedding.from_module(embedding, bits=4, symmetric=True, group_size=32)

        x = torch.randint(0, 1000, (4, 8, 32))
        output = qembed(x)

        assert output.shape == (4, 8, 32, 256)

    def test_from_tensors_with_precomputed_params(self):
        """Test creating QuantEmbedding with precomputed scales and zero points."""
        weight = torch.randn(1000, 256)

        # Compute quantization parameters
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32, signed=False)
        scales, zero_points = quantizer.find_qparams(weight)

        # Create QuantEmbedding with precomputed params
        qembed = QuantEmbedding.from_tensors(
            num_embeddings=1000,
            embedding_dim=256,
            weight=weight,
            bits=4,
            symmetric=True,
            group_size=32,
            scales=scales,
            zero_points=zero_points,
        )

        assert qembed.scales.shape == scales.shape
        assert torch.all(qembed.scales == scales)

    def test_from_tensors_quantized_weights(self):
        """Test creating QuantEmbedding from already quantized weights."""
        weight = torch.randn(1000, 256)

        # Quantize weights
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32, signed=False)
        scales, zero_points = quantizer.find_qparams(weight)
        qweight = quantizer.quantize(weight, scales, zero_points)

        # Create QuantEmbedding from quantized weights
        qembed = QuantEmbedding.from_tensors(
            num_embeddings=1000,
            embedding_dim=256,
            weight=qweight,
            bits=4,
            symmetric=True,
            group_size=32,
            scales=scales,
            zero_points=zero_points,
            quantized=True,
        )

        assert qembed.rows == 1000
        assert qembed.cols == 256

    def test_extra_repr(self):
        """Test string representation of QuantEmbedding."""
        qembed = QuantEmbedding(
            num_embeddings=1000,
            embedding_dim=256,
            bits=4,
            symmetric=True,
            group_size=32,
            padding_idx=0,
        )

        repr_str = qembed.extra_repr()
        assert "1000" in repr_str
        assert "256" in repr_str
        assert "bits=4" in repr_str
        assert "symmetric=True" in repr_str
        assert "group_size=32" in repr_str
        assert "padding_idx=0" in repr_str


class TestQuantModuleIntegration:
    def test_quantlinear_forward_backward_compatibility(self):
        """Test that QuantLinear produces similar results to nn.Linear."""
        # Create a linear layer
        linear = nn.Linear(128, 64, bias=True)
        linear.weight.data.normal_(0, 0.02)
        linear.bias.data.zero_()

        # Create quantized version
        qlinear = QuantLinear.from_module(linear, bits=8, symmetric=True, group_size=32)

        # Test with same input
        x = torch.randn(16, 128)

        # Forward pass
        linear_out = linear(x)
        qlinear_out = qlinear(x)

        # Outputs should be close (not exact due to quantization)
        # For 8-bit, we expect decent accuracy
        max_diff = torch.max(torch.abs(linear_out - qlinear_out))
        relative_error = max_diff / torch.max(torch.abs(linear_out))
        assert relative_error < 0.1, f"Relative error {relative_error} too large"

    def test_quantembedding_forward_backward_compatibility(self):
        """Test that QuantEmbedding produces similar results to nn.Embedding."""
        # Create an embedding layer
        embedding = nn.Embedding(100, 64)
        embedding.weight.data.normal_(0, 0.02)

        # Create quantized version
        qembed = QuantEmbedding.from_module(embedding, bits=8, symmetric=True, group_size=16)

        # Test with same input
        x = torch.randint(0, 100, (16,))

        # Forward pass
        embed_out = embedding(x)
        qembed_out = qembed(x)

        # Outputs should be close (not exact due to quantization)
        max_diff = torch.max(torch.abs(embed_out - qembed_out))
        relative_error = max_diff / torch.max(torch.abs(embed_out))
        assert relative_error < 0.1, f"Relative error {relative_error} too large"

    def test_device_placement(self):
        """Test that QuantLinear respects device placement."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        device = torch.device("cuda")
        qlinear = QuantLinear(
            in_features=128,
            out_features=256,
            bits=4,
            symmetric=True,
            group_size=32,
            device=device,
        )

        assert qlinear.qweight.device.type == device.type
        assert qlinear.scales.device.type == device.type
        if qlinear.bias is not None:
            assert qlinear.bias.device.type == device.type

    def test_dtype_consistency(self):
        """Test that QuantLinear maintains dtype consistency."""
        dtype = torch.float16
        qlinear = QuantLinear(
            in_features=128,
            out_features=256,
            bits=4,
            symmetric=True,
            group_size=32,
            dtype=dtype,
        )

        assert qlinear.scales.dtype == dtype
        if qlinear.bias is not None:
            assert qlinear.bias.dtype == dtype
