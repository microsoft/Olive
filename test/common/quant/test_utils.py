# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.common.quant.utils import WeightQuantizer, get_maxq_minq, pack_to_uint8, unpack_from_uint8

# pylint: disable=W0212


class TestGetMaxqMinq:
    def test_unsigned_4bit(self):
        """Test 4-bit unsigned quantization range."""
        maxq, minq = get_maxq_minq(bits=4, signed=False)
        assert minq == 0
        assert maxq == 15
        assert maxq - minq + 1 == 16

    def test_unsigned_8bit(self):
        """Test 8-bit unsigned quantization range."""
        maxq, minq = get_maxq_minq(bits=8, signed=False)
        assert minq == 0
        assert maxq == 255
        assert maxq - minq + 1 == 256

    def test_signed_4bit(self):
        """Test 4-bit signed quantization range."""
        maxq, minq = get_maxq_minq(bits=4, signed=True)
        assert minq == -8
        assert maxq == 7
        assert maxq - minq + 1 == 16

    def test_signed_8bit(self):
        """Test 8-bit signed quantization range."""
        maxq, minq = get_maxq_minq(bits=8, signed=True)
        assert minq == -128
        assert maxq == 127
        assert maxq - minq + 1 == 256


class TestWeightQuantizer:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("group_size", [0, 16, -1])
    @pytest.mark.parametrize("signed", [True, False])
    def test_initialization(self, bits, symmetric, group_size, signed):
        """Test WeightQuantizer initialization with various parameters."""
        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=signed)
        assert quantizer.bits == bits
        assert quantizer.symmetric == symmetric
        assert quantizer.group_size == group_size
        assert quantizer.signed == signed

        maxq, minq = quantizer.maxq, quantizer.minq
        if signed:
            assert minq < 0
        else:
            assert minq >= 0
        assert maxq > minq
        assert maxq - minq + 1 == 2**bits

    def test_invalid_bits(self):
        """Test that invalid bits raise AssertionError."""
        with pytest.raises(AssertionError, match="Only 4-bit and 8-bit quantization supported"):
            WeightQuantizer(bits=16, symmetric=True, group_size=0)

    def test_midq_calculation(self):
        """Test that midq is calculated correctly."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=0, signed=False)
        # For unsigned 4-bit: minq=0, maxq=15, midq should be 8
        assert quantizer.midq == 8

        quantizer_signed = WeightQuantizer(bits=4, symmetric=True, group_size=0, signed=True)
        # For signed 4-bit: minq=-8, maxq=7, midq should be 0
        assert quantizer_signed.midq == 0

    @pytest.mark.parametrize("group_size", [0, 16, 32, -1])
    def test_get_num_groups(self, group_size):
        """Test get_num_groups for different group sizes."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=group_size)
        shape = (64, 128)

        if group_size == 0:
            with pytest.raises(ValueError, match="group_size must be greater than 0"):
                quantizer.get_num_groups(shape)
        elif group_size == -1:
            # Per-channel quantization: one group per row
            num_groups = quantizer.get_num_groups(shape)
            assert num_groups == 1
        else:
            num_groups = quantizer.get_num_groups(shape)
            assert num_groups == shape[1] // group_size

    def test_get_num_groups_invalid_divisibility(self):
        """Test that get_num_groups fails when shape is not divisible by group_size."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32)
        shape = (64, 100)  # 100 is not divisible by 32

        with pytest.raises(AssertionError, match="in_features .* must be divisible by group_size"):
            quantizer.get_num_groups(shape)

    @pytest.mark.parametrize("group_size", [0, 16, -1])
    def test_get_qparam_shape(self, group_size):
        """Test get_qparam_shape for different group sizes."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=group_size)
        shape = (64, 128)

        qparam_shape = quantizer.get_qparam_shape(shape)

        if group_size == 0:
            assert qparam_shape == (1, 1)  # Per-tensor
        elif group_size == -1:
            assert qparam_shape == (64, 1)  # Per-channel
        else:
            expected_num_groups = shape[1] // group_size
            assert qparam_shape == (64, expected_num_groups)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("group_size", [0, 16, -1])
    def test_find_qparams(self, bits, symmetric, group_size):
        """Test finding quantization parameters."""
        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        weight = torch.randn(32, 64)

        scales, zero_points = quantizer.find_qparams(weight)

        # Check dtypes
        assert scales.dtype == weight.dtype
        assert zero_points.dtype == torch.int32

        # Check shapes
        expected_shape = quantizer.get_qparam_shape(weight.shape)
        assert scales.shape == expected_shape
        assert zero_points.shape == expected_shape

        # Check that scales are positive
        assert torch.all(scales > 0)

        # Check that zero points are within range
        assert torch.all(zero_points >= quantizer.minq)
        assert torch.all(zero_points <= quantizer.maxq)

        if symmetric:
            # For symmetric quantization, zero points should be at midq
            assert torch.all(zero_points == quantizer.midq)

    def test_find_qparams_zero_tensor(self):
        """Test finding quantization parameters for zero tensor."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=0, signed=False)
        weight = torch.zeros(32, 64)

        scales, _ = quantizer.find_qparams(weight)

        # Should handle zero tensor gracefully
        assert torch.all(scales > 0)
        assert not torch.any(torch.isnan(scales))
        assert not torch.any(torch.isinf(scales))

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("group_size", [0, 16, -1])
    def test_quantize(self, bits, symmetric, group_size):
        """Test quantization of weights."""
        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        weight = torch.randn(32, 64)

        scales, zero_points = quantizer.find_qparams(weight)
        qweight = quantizer.quantize(weight, scales, zero_points)

        # Check dtype and shape
        assert qweight.dtype == torch.int32
        assert qweight.shape == weight.shape

        # Check that quantized values are within range
        assert torch.all(qweight >= quantizer.minq)
        assert torch.all(qweight <= quantizer.maxq)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_dequantize(self, bits):
        """Test dequantization of weights."""
        quantizer = WeightQuantizer(bits=bits, symmetric=True, group_size=32, signed=False)
        weight = torch.randn(32, 64)

        # Quantize
        scales, zero_points = quantizer.find_qparams(weight)
        qweight = quantizer.quantize(weight, scales, zero_points)

        # Dequantize
        dq_weight = quantizer.dequantize(qweight, scales, zero_points)

        # Check dtype and shape
        assert dq_weight.dtype == weight.dtype
        assert dq_weight.shape == weight.shape

    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("symmetric", [True, False])
    @pytest.mark.parametrize("group_size", [0, 16, -1])
    def test_fake_quantize(self, bits, symmetric, group_size):
        """Test fake quantization (quantize then dequantize)."""
        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        weight = torch.randn(32, 64)

        scales, zero_points = quantizer.find_qparams(weight)
        qdq_weight = quantizer.fake_quantize(weight, scales, zero_points)

        # Check dtype and shape
        assert qdq_weight.dtype == weight.dtype
        assert qdq_weight.shape == weight.shape

        # Verify it matches quantize->dequantize
        qweight = quantizer.quantize(weight, scales, zero_points)
        dq_weight = quantizer.dequantize(qweight, scales, zero_points)
        assert torch.all(dq_weight == qdq_weight)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_quantization_accuracy(self, bits):
        """Test that quantization error is within expected bounds."""
        quantizer = WeightQuantizer(bits=bits, symmetric=True, group_size=32, signed=False)
        weight = torch.randn(32, 64) * 0.02

        scales, zero_points = quantizer.find_qparams(weight)
        qdq_weight = quantizer.fake_quantize(weight, scales, zero_points)

        # Calculate quantization error
        max_abs_error = torch.max(torch.abs(weight - qdq_weight))
        relative_error = max_abs_error / torch.max(torch.abs(weight))

        # Error should be reasonable for the bit width
        tolerance = 0.2 if bits == 4 else 0.1 if bits == 8 else 0.4
        assert relative_error < tolerance, f"Relative error {relative_error} exceeds tolerance {tolerance}"

    def test_reshape_tensor_per_tensor(self):
        """Test tensor reshaping for per-tensor quantization."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=0, signed=False)
        tensor = torch.randn(32, 64)

        reshaped, original_shape = quantizer._reshape_tensor(tensor)

        assert reshaped.shape == (1, 1, 32 * 64)
        assert original_shape == (32, 64)

    def test_reshape_tensor_per_channel(self):
        """Test tensor reshaping for per-channel quantization."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=-1, signed=False)
        tensor = torch.randn(32, 64)

        reshaped, original_shape = quantizer._reshape_tensor(tensor)

        assert reshaped.shape == (32, 1, 64)
        assert original_shape == (32, 64)

    def test_reshape_tensor_groupwise(self):
        """Test tensor reshaping for groupwise quantization."""
        quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=16, signed=False)
        tensor = torch.randn(32, 64)

        reshaped, original_shape = quantizer._reshape_tensor(tensor)

        assert reshaped.shape == (32, 4, 16)  # 64 / 16 = 4 groups
        assert original_shape == (32, 64)


class TestPackUnpack:
    @pytest.mark.parametrize("bits", [2, 4, 8])
    @pytest.mark.parametrize("shape", [(16, 16), (16, 1), (32, 64), (1, 128)])
    def test_pack_unpack_round_trip(self, bits, shape):
        """Test that packing and unpacking preserves values."""
        tensor = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
        packed = pack_to_uint8(tensor, bits)
        unpacked = unpack_from_uint8(packed, bits, shape)
        assert torch.all(tensor == unpacked)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_pack_shape(self, bits):
        """Test that packed tensor has correct shape."""
        shape = (32, 64)
        tensor = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
        packed = pack_to_uint8(tensor, bits)

        packing_factor = 8 // bits
        expected_packed_cols = (shape[1] + packing_factor - 1) // packing_factor
        assert packed.shape == (shape[0], expected_packed_cols)
        assert packed.dtype == torch.uint8

    def test_pack_invalid_bits(self):
        """Test that packing with invalid bits raises error."""
        tensor = torch.randint(0, 16, (16, 16), dtype=torch.uint8)

        # Bits must be 2, 4 or 8
        with pytest.raises(AssertionError, match="Only 2-bit, 4-bit and 8-bit quantization supported"):
            pack_to_uint8(tensor, bits=16)

    def test_pack_values_out_of_range_high(self):
        """Test that packing with values exceeding max raises error."""
        tensor = torch.randint(0, 32, (16, 16), dtype=torch.uint8)  # Values up to 31

        # For 4-bit, max is 15
        with pytest.raises(AssertionError, match="Input tensor values must not exceed max quantization value"):
            pack_to_uint8(tensor, bits=4)

    def test_unpack_dtype_validation(self):
        """Test that unpacking requires uint8 input."""
        tensor = torch.randint(0, 16, (16, 8), dtype=torch.int32)

        with pytest.raises(AssertionError, match="Input tensor must be of dtype uint8"):
            unpack_from_uint8(tensor, bits=4, shape=(16, 16))

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_pack_with_padding(self, bits):
        """Test packing with shapes that require padding."""
        # Create a shape that doesn't divide evenly
        cols = 17  # Not divisible by packing_factor for 4-bit
        shape = (8, cols)
        tensor = torch.randint(0, 2**bits, shape, dtype=torch.uint8)

        packed = pack_to_uint8(tensor, bits)
        unpacked = unpack_from_uint8(packed, bits, shape)

        # Should still work correctly
        assert torch.all(tensor == unpacked)

    def test_unpack_shape_trimming(self):
        """Test that unpacking trims to the correct shape."""
        bits = 4
        original_shape = (8, 17)  # Not divisible by packing factor
        tensor = torch.randint(0, 2**bits, original_shape, dtype=torch.uint8)

        packed = pack_to_uint8(tensor, bits)
        unpacked = unpack_from_uint8(packed, bits, original_shape)

        assert unpacked.shape == original_shape
        assert torch.all(tensor == unpacked)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_pack_all_zeros(self, bits):
        """Test packing tensor with all zeros."""
        shape = (16, 32)
        tensor = torch.zeros(shape, dtype=torch.uint8)

        packed = pack_to_uint8(tensor, bits)
        unpacked = unpack_from_uint8(packed, bits, shape)

        assert torch.all(unpacked == 0)

    @pytest.mark.parametrize("bits", [2, 4, 8])
    def test_pack_max_values(self, bits):
        """Test packing tensor with maximum values."""
        shape = (16, 32)
        max_val = (1 << bits) - 1
        tensor = torch.full(shape, max_val, dtype=torch.uint8)

        packed = pack_to_uint8(tensor, bits)
        unpacked = unpack_from_uint8(packed, bits, shape)

        assert torch.all(unpacked == max_val)

    def test_pack_4bit_specific_pattern(self):
        """Test packing with specific 4-bit pattern to verify bit manipulation."""
        # Create a tensor with known pattern
        tensor = torch.tensor([[0, 1, 2, 3, 4, 5, 6, 7]], dtype=torch.uint8)
        packed = pack_to_uint8(tensor, bits=4)

        # For 4-bit, two values pack into one byte
        # Expected: [0x10, 0x32, 0x54, 0x76]
        assert packed.shape == (1, 4)
        expected = torch.tensor([[0x10, 0x32, 0x54, 0x76]], dtype=torch.uint8)
        assert torch.all(packed == expected)

        # Verify unpacking
        unpacked = unpack_from_uint8(packed, bits=4, shape=tensor.shape)
        assert torch.all(unpacked == tensor)

    def test_unpack_output_dtype(self):
        """Test that unpacked tensor has int32 dtype."""
        tensor = torch.randint(0, 16, (8, 16), dtype=torch.uint8)
        packed = pack_to_uint8(tensor, bits=4)
        unpacked = unpack_from_uint8(packed, bits=4, shape=tensor.shape)

        assert unpacked.dtype == torch.int32

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")),
        ],
    )
    def test_pack_unpack_device_consistency(self, device):
        """Test that packing and unpacking work on different devices."""
        device = torch.device(device)
        tensor = torch.randint(0, 16, (16, 32), dtype=torch.uint8, device=device)

        packed = pack_to_uint8(tensor, bits=4)
        assert packed.device.type == device.type

        unpacked = unpack_from_uint8(packed, bits=4, shape=tensor.shape)
        assert unpacked.device.type == device.type
        assert torch.all(unpacked == tensor.to(torch.int32))
