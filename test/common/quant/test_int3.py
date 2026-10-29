import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig, _build_placeholder_quant_tensor
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer, pack_to_uint8, unpack_from_uint8


@pytest.mark.parametrize("count", [1, 3, 7, 8, 9, 31, 32, 33, 129])
def test_int3_row_local_packing_matches_scalar_oracle(count):
    codes = torch.arange(5 * count).reshape(5, count) % 8
    expected = torch.zeros(5, (count * 3 + 7) // 8, dtype=torch.uint8)
    for row in range(5):
        for element in range(count):
            for bit in range(3):
                position = element * 3 + bit
                expected[row, position // 8] |= ((int(codes[row, element]) >> bit) & 1) << (position % 8)
    actual = pack_to_uint8(codes, 3)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(unpack_from_uint8(actual, 3, tuple(codes.shape)), codes.to(torch.int32), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("block", [16, 32, 64, 128])
def test_int3_quant_tensor_uses_stored_scale(dtype, block):
    weight = ((torch.arange(2 * 5 * block * 3).reshape(2, 5, block * 3) % 31 - 15) * 0.013).to(dtype)
    quantized = QuantTensor.from_float(weight, bits=3, group_size=block, symmetric=True)
    groups = weight.reshape(2, 5, 3, block)
    scales = groups.abs().amax(-1) / 3.5
    compute_dtype = torch.promote_types(dtype, torch.float32)
    expected_codes = (
        (torch.round(groups.to(compute_dtype) / scales.to(compute_dtype).unsqueeze(-1)) + 4)
        .clamp(0, 7)
        .reshape(weight.shape)
        .to(torch.int32)
    )
    actual_codes = unpack_from_uint8(quantized.qweight, 3, tuple(weight.shape))
    torch.testing.assert_close(quantized.scales, scales, rtol=0, atol=0)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    assert quantized.qzeros is None


def test_int3_existing_scale_policy_matches_golden_packed_codes():
    weight = torch.tensor([[-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]]).repeat(1, 4)
    quantized = QuantTensor.from_float(weight, bits=3, group_size=32, symmetric=True)
    torch.testing.assert_close(quantized.scales, torch.ones(1, 1), rtol=0, atol=0)
    expected_codes = torch.tensor([[0, 2, 2, 4, 4, 6, 6, 7]], dtype=torch.int32).repeat(1, 4)
    expected_bytes = torch.tensor([[0x90, 0x48, 0xFB]], dtype=torch.uint8).repeat(1, 4)
    torch.testing.assert_close(quantized.qweight, expected_bytes, rtol=0, atol=0)
    torch.testing.assert_close(
        unpack_from_uint8(quantized.qweight, 3, tuple(weight.shape)), expected_codes, rtol=0, atol=0
    )
    torch.testing.assert_close(quantized.to_dense(), (expected_codes - 4).float(), rtol=0, atol=0)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("group_size", [0, -1, 16, 32, 64, 128])
def test_int3_zero_blocks_preserve_existing_scale(dtype, group_size):
    zero = torch.zeros(2, 4, 128, dtype=dtype)
    quantized = QuantTensor.from_float(zero, bits=3, group_size=group_size, symmetric=True)
    torch.testing.assert_close(quantized.scales, torch.full_like(quantized.scales, 1 / 3.5), rtol=0, atol=0)
    assert torch.all(unpack_from_uint8(quantized.qweight, 3, tuple(zero.shape)) == 4)
    torch.testing.assert_close(quantized.to_dense(), zero, rtol=0, atol=0)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("group_size", [0, -1, 16, 32, 64, 128])
def test_int3_rejects_nonfinite_weights(value, group_size):
    weight = torch.zeros(2, 128)
    weight[1, 0] = value
    with pytest.raises(ValueError, match="finite"):
        QuantTensor.from_float(weight, bits=3, group_size=group_size, symmetric=True)


def test_int3_rounds_quotient_before_adding_zero_point():
    weight = torch.zeros(1, 32, dtype=torch.float32)
    weight[0, :4] = torch.tensor([3.5, -3.5, 0.50000006, -0.50000006])
    quantized = QuantTensor.from_float(weight, bits=3, group_size=32, symmetric=True)
    codes = unpack_from_uint8(quantized.qweight, 3, tuple(weight.shape))
    torch.testing.assert_close(codes[0, :4], torch.tensor([7, 0, 5, 3], dtype=torch.int32))


@pytest.mark.parametrize("block", [32, 64, 128])
def test_int3_preserves_float64_scale_precision(block):
    weight = torch.zeros(2, block, dtype=torch.float64)
    weight[:, :2] = torch.tensor([3.5e100, -2e100], dtype=torch.float64)
    quantized = QuantTensor.from_float(weight, bits=3, group_size=block, symmetric=True)
    torch.testing.assert_close(quantized.scales, torch.full((2, 1), 1e100, dtype=torch.float64), rtol=0, atol=0)
    expected_codes = torch.full(weight.shape, 4, dtype=torch.int32)
    expected_codes[:, :2] = torch.tensor([7, 2], dtype=torch.int32)
    torch.testing.assert_close(unpack_from_uint8(quantized.qweight, 3, tuple(weight.shape)), expected_codes)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("sign", [-1, 1])
@pytest.mark.parametrize("group_size", [0, -1, 16, 32, 64, 128])
def test_int3_rejects_scale_underflow_during_computation(dtype, sign, group_size):
    smallest = torch.nextafter(torch.tensor(0, dtype=dtype), torch.tensor(1, dtype=dtype))
    weight = torch.zeros(2, 128, dtype=dtype)
    weight[1, 0] = sign * smallest
    with pytest.raises(ValueError, match="underflow"):
        QuantTensor.from_float(weight, bits=3, group_size=group_size, symmetric=True)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("group_size", [0, -1, 16, 32, 64, 128])
@pytest.mark.parametrize("sign", [-1, 1])
def test_int3_nonzero_scales_preserve_existing_policy(dtype, group_size, sign):
    weight = torch.linspace(0, 1, 256, dtype=dtype).reshape(2, 128) * sign
    quantizer = WeightQuantizer(bits=3, symmetric=True, group_size=group_size)
    scales, zeros = quantizer.find_qparams(weight)
    if group_size == 0:
        bound = weight.abs().amax().reshape(1, 1)
    else:
        block = weight.shape[-1] if group_size == -1 else group_size
        bound = weight.reshape(2, -1, block).abs().amax(-1)
    torch.testing.assert_close(scales, bound / 3.5, rtol=0, atol=0)
    assert torch.all(zeros == 4)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
def test_int3_large_finite_weights_preserve_existing_scale(dtype):
    weight = torch.full((2, 32), torch.finfo(dtype).max, dtype=dtype)
    weight[1].neg_()
    quantized = QuantTensor.from_float(weight, bits=3, group_size=32, symmetric=True)
    torch.testing.assert_close(quantized.scales, weight.abs().amax(-1, keepdim=True) / 3.5, rtol=0, atol=0)
    assert torch.isfinite(quantized.scales).all()


def test_int3_expert_override_and_checkpoint_placeholder():
    config = OliveHfQuantizationConfig(
        bits=4,
        symmetric=True,
        group_size=64,
        moe=True,
        overrides={"re:model\\.layers\\.\\d+\\.mlp\\.experts\\.(?:gate_up_proj|down_proj)": {"bits": 3}},
    )
    assert config.get_qlinear_init_args("model.layers.0.mlp.experts.gate_up_proj")["bits"] == 3
    assert config.get_qlinear_init_args("model.layers.0.mlp.experts.down_proj")["bits"] == 3
    assert config.get_qlinear_init_args("model.layers.0.self_attn.q_proj")["bits"] == 4
    weight = torch.zeros(2, 192, 64, dtype=torch.float16)
    actual = QuantTensor.from_float(weight, bits=3, group_size=64, symmetric=True)
    placeholder = _build_placeholder_quant_tensor(
        shape=tuple(weight.shape),
        bits=3,
        symmetric=True,
        group_size=64,
        dtype=weight.dtype,
        device=weight.device,
    )
    assert placeholder.qweight.shape == actual.qweight.shape == (2, 192, 24)
    assert placeholder.scales.shape == actual.scales.shape
