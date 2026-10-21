import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig, _build_placeholder_quant_tensor
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import pack_to_uint8, unpack_from_uint8


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


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("block", [32, 64, 128])
def test_int3_quant_tensor_uses_stored_scale(dtype, block):
    weight = ((torch.arange(2 * 5 * block * 3).reshape(2, 5, block * 3) % 31 - 15) * 0.013).to(dtype)
    quantized = QuantTensor.from_float(weight, bits=3, group_size=block, symmetric=True)
    groups = weight.float().reshape(2, 5, 3, block)
    scales = torch.maximum((-groups).clamp_min(0).amax(-1) / 4, groups.clamp_min(0).amax(-1) / 3).to(dtype)
    expected_codes = (
        torch.round(groups / scales.float().unsqueeze(-1) + 4).clamp(0, 7).reshape(weight.shape).to(torch.int32)
    )
    actual_codes = unpack_from_uint8(quantized.qweight, 3, tuple(weight.shape))
    torch.testing.assert_close(quantized.scales, scales, rtol=0, atol=0)
    torch.testing.assert_close(actual_codes, expected_codes, rtol=0, atol=0)
    assert quantized.qzeros is None


def test_int3_zero_blocks_and_invalid_weights():
    zero = torch.zeros(2, 4, 32, dtype=torch.float16)
    quantized = QuantTensor.from_float(zero, bits=3, group_size=32, symmetric=True)
    assert torch.all(quantized.scales == 0)
    assert torch.all(unpack_from_uint8(quantized.qweight, 3, tuple(zero.shape)) == 4)
    with pytest.raises(ValueError, match="finite"):
        QuantTensor.from_float(torch.full_like(zero, float("nan")), bits=3, group_size=32, symmetric=True)
    tiny = torch.full_like(zero, torch.finfo(torch.float16).smallest_normal / 1024)
    with pytest.raises(ValueError, match="underflow"):
        QuantTensor.from_float(tiny, bits=3, group_size=32, symmetric=True)


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
