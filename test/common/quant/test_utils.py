# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch

from olive.common.quant.utils import WeightQuantizer, pack_to_uint8, unpack_from_uint8


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("group_size", [0, 16, -1])
@pytest.mark.parametrize("signed", [True, False])
def test_weight_quantizer(bits, symmetric, group_size, signed):
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

    weight = torch.randn(32, 64)
    scales, zero_points = quantizer.find_qparams(weight)
    assert scales.dtype == weight.dtype
    assert zero_points.dtype == torch.int32

    qweight = quantizer.quantize(weight, scales, zero_points)
    assert qweight.dtype == torch.int32

    dq_weight = quantizer.dequantize(qweight, scales, zero_points)
    assert dq_weight.dtype == weight.dtype
    assert dq_weight.shape == weight.shape

    qdq_weight = quantizer.fake_quantize(weight, scales, zero_points)
    assert torch.all(dq_weight == qdq_weight)


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("shape", [(16, 16), (16, 1)])
def test_pack_unpack_uint8(bits, shape):
    tensor = torch.randint(0, 2**bits, shape, dtype=torch.uint8)
    packed = pack_to_uint8(tensor, bits)
    unpacked = unpack_from_uint8(packed, bits, shape)
    assert torch.all(tensor == unpacked)
