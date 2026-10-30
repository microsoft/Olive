# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import pytest
import torch

from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.passes.pytorch.kquant import _kquant_search, kquant_find_qparams


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("signed", [False, True])
@pytest.mark.parametrize("group_size", [0, -1, 32])
def test_symmetric_nonnegative_groups_preserve_scale_under_sign_flip(dtype, bits, signed, group_size):
    quantizer = WeightQuantizer(bits=bits, symmetric=True, group_size=group_size, signed=signed)
    weight = torch.linspace(0, 1, 64).reshape(2, 1, 32).to(dtype)
    positive_scales, positive_zeros = quantizer.find_qparams(weight)
    negative_scales, negative_zeros = quantizer.find_qparams(-weight)
    assert torch.equal(positive_scales, negative_scales)
    assert torch.equal(positive_zeros, negative_zeros)
    assert positive_scales.dtype == dtype


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_symmetric_scale_avoids_intermediate_range_overflow(dtype):
    quantizer = WeightQuantizer(bits=4, symmetric=True, group_size=32)
    positive = torch.full((1, 32), torch.finfo(dtype).max * 0.625, dtype=dtype)
    weight = torch.cat((positive, -positive))
    scales, zeros = quantizer.find_qparams(weight)
    assert torch.isfinite(scales).all()
    assert torch.isfinite(quantizer.fake_quantize(weight, scales, zeros)).all()
    assert torch.equal(scales, torch.full_like(scales, positive[0, 0].item() / 7.5))


@pytest.mark.parametrize(
    ("dtype", "weight_value", "scale_value"),
    [
        (torch.bfloat16, 0.0167236328125, 0.01116943359375),
        (torch.float16, 1.4990234375, 0.99951171875),
    ],
)
def test_quantize_promotes_division_before_rounding(dtype, weight_value, scale_value):
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=-1)
    weight = torch.tensor([[weight_value]], dtype=dtype)
    scales = torch.tensor([[scale_value]], dtype=dtype)
    zeros = torch.full((1, 1), 6, dtype=torch.int32)
    assert (weight / scales).item() == 1.5
    assert (weight.float() / scales.float()).item() < 1.5
    assert quantizer.quantize(weight, scales, zeros).item() == 7


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_quantize_low_precision_avoids_false_rounding_ties(dtype):
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=32)
    below_half = torch.nextafter(torch.tensor(1.5, dtype=dtype), torch.tensor(0.0, dtype=dtype))
    weight = below_half.expand(2, 3, 32)
    scales = torch.ones(2, 3, 1, dtype=dtype)
    zeros = torch.full((2, 3, 1), 6, dtype=torch.int32)
    assert torch.equal(quantizer.quantize(weight, scales, zeros), torch.full_like(weight, 7, dtype=torch.int32))


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_quantize_rounds_before_adding_zero_point(dtype):
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=-1)
    weight = torch.tensor([[0.5, 1.5, 2.5, -0.5, -1.5, -2.5]], dtype=dtype)
    scales = torch.ones(1, 1, dtype=dtype)
    zeros = torch.full((1, 1), 5, dtype=torch.int32)
    assert torch.equal(quantizer.quantize(weight, scales, zeros), torch.tensor([[5, 7, 7, 5, 3, 3]], dtype=torch.int32))


def test_quantize_does_not_narrow_float64_scales():
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=-1)
    weight = torch.tensor([[1.5]], dtype=torch.float32)
    scales = torch.tensor([[1 + 2**-30]], dtype=torch.float64)
    zeros = torch.full((1, 1), 3, dtype=torch.int32)
    assert quantizer.quantize(weight, scales, zeros).item() == 4
    assert quantizer.quantize(weight, scales.float(), zeros).item() == 5


def _real_group_weight(dtype):
    return torch.tensor(
        [
            [
                0.427734375,
                0.04833984375,
                0.07666015625,
                -0.03515625,
                -0.06640625,
                -0.010986328125,
                0.15625,
                0.416015625,
                0.1064453125,
                -0.0927734375,
                0.047607421875,
                -0.2578125,
                0.0140380859375,
                -0.00848388671875,
                0.08837890625,
                0.08544921875,
                0.1591796875,
                0.5390625,
                -0.02001953125,
                -0.1689453125,
                -0.181640625,
                -0.09912109375,
                -0.34765625,
                0.0279541015625,
                0.03857421875,
                0.08349609375,
                0.2275390625,
                0.123046875,
                0.046875,
                -0.07763671875,
                0.486328125,
                0.166015625,
            ]
        ],
        dtype=dtype,
    )


def _stored_weighted_group_error(weight, scales, zeros, bits, symmetric):
    quantized = QuantTensor.from_float(
        weight, bits=bits, symmetric=symmetric, group_size=32, scales=scales, zero_points=zeros
    )
    grouped = weight.float().reshape(-1, 32)
    importance = grouped.square().mean(-1, keepdim=True).sqrt() + grouped.abs()
    error = quantized.to_dense().double().reshape(-1, 32) - weight.double().reshape(-1, 32)
    return (importance.double() * error.square()).sum(-1)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_kquant_real_group_scores_integer_affine_storage(dtype):
    weight = _real_group_weight(dtype)
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=32)
    rtn_scales, rtn_zeros = quantizer.find_qparams(weight)
    scales, zeros = kquant_find_qparams(weight, 32, 15, 0)
    assert torch.all(
        _stored_weighted_group_error(weight, scales, zeros, 4, False)
        <= _stored_weighted_group_error(weight, rtn_scales, rtn_zeros, 4, False)
    )
    assert scales.dtype == dtype
    assert zeros.dtype == torch.int32


def test_kquant_search_selects_stored_error_winner_not_free_offset_winner():
    weight = _real_group_weight(torch.bfloat16)
    data = weight.float()
    importance = data.square().mean(-1, keepdim=True).sqrt() + data.abs()
    scale_a, offset_a = torch.tensor([[0.0634688138961792]]), torch.tensor([[-0.3441556692123413]])
    scale_b = torch.tensor([[0.0595703125]])
    offset_b = -6 * scale_b

    def free_error(scale, offset):
        codes = ((data - offset) / scale).round().clamp(0, 15)
        return (importance * (scale * codes + offset - data).square()).sum(-1, keepdim=True)

    def stored_error(scale, offset):
        zeros = (-offset / scale).round().clamp(0, 15).int()
        return _stored_weighted_group_error(weight, scale.to(weight.dtype), zeros, 4, False).reshape(-1, 1)

    assert free_error(scale_a, offset_a).item() < free_error(scale_b, offset_b).item()
    assert stored_error(scale_b, offset_b).item() < stored_error(scale_a, offset_a).item()
    scale, offset = _kquant_search(
        data,
        importance,
        torch.ones_like(scale_a),
        offset_a,
        0,
        15,
        initial_factor=1 / scale_a.item(),
        candidate_factors=(1.0,),
        refine_fn=lambda *_: (scale_b, offset_b),
        score_fn=stored_error,
    )
    assert torch.equal(scale, scale_b)
    assert torch.equal(offset, offset_b)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("value", [1e20, -1e20])
def test_kquant_finite_large_groups_do_not_overflow_scoring(dtype, value):
    weight = torch.full((1, 32), value, dtype=dtype)
    scales, zeros = kquant_find_qparams(weight, 32, 15, 0)
    quantizer = WeightQuantizer(bits=4, symmetric=False, group_size=32)
    assert torch.isfinite(scales).all()
    assert (scales > 0).all()
    assert torch.isfinite(quantizer.fake_quantize(weight, scales, zeros)).all()


@pytest.mark.parametrize("factor", [2.0**-80, 2.0**80])
def test_kquant_search_preserves_power_of_two_rescaling(factor):
    weight = torch.randn(2, 64, generator=torch.Generator().manual_seed(42))
    scales, zeros = kquant_find_qparams(weight, 32, 15, 0)
    scaled_scales, scaled_zeros = kquant_find_qparams(weight * factor, 32, 15, 0)
    torch.testing.assert_close(scaled_scales / factor, scales, rtol=0, atol=0)
    torch.testing.assert_close(scaled_zeros, zeros, rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("bits", [2, 3, 4, 8])
@pytest.mark.parametrize("symmetric", [False, True])
@pytest.mark.parametrize("shape", [(4, 64), (2, 3, 64)])
def test_kquant_stored_group_error_is_bounded_by_rtn(dtype, bits, symmetric, shape):
    weight = torch.randn(shape, generator=torch.Generator().manual_seed(42)).to(dtype)
    quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=32)
    rtn_scales, rtn_zeros = quantizer.find_qparams(weight)
    scales, zeros = kquant_find_qparams(weight, 32, quantizer.maxq, quantizer.minq, symmetric)
    actual = _stored_weighted_group_error(weight, scales, zeros, bits, symmetric)
    baseline = _stored_weighted_group_error(weight, rtn_scales, rtn_zeros, bits, symmetric)
    assert torch.all(actual <= baseline * (1 + 1e-6) + 1e-12)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_kquant_rejects_nonfinite_groups(value):
    with pytest.raises(ValueError, match="finite quantization parameters"):
        kquant_find_qparams(torch.full((1, 32), value), 32, 15, 0)


def test_kquant_retains_upstream_subnormal_scale_floor():
    smallest = torch.nextafter(torch.tensor(0.0, dtype=torch.float16), torch.tensor(1.0, dtype=torch.float16))
    weight = smallest.expand(1, 32)
    scales, zeros = kquant_find_qparams(weight, 32, 15, 0)
    assert torch.equal(scales, smallest.reshape(1, 1))
    assert torch.isfinite(WeightQuantizer(group_size=32).fake_quantize(weight, scales, zeros)).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("chunk_elements", [32, 64, 128])
def test_kquant_chunking_preserves_storage_aware_parameters(dtype, chunk_elements):
    weight = torch.randn(2, 3, 64, generator=torch.Generator().manual_seed(42)).to(dtype)
    expected = kquant_find_qparams(weight, 32, 15, 0)
    actual = kquant_find_qparams(weight, 32, 15, 0, device="cpu", max_chunk_elements=chunk_elements)
    for chunked, unchunked in zip(actual, expected):
        torch.testing.assert_close(chunked, unchunked, rtol=0, atol=0)
