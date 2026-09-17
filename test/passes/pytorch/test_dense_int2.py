# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch

from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import get_maxq_minq
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.gptq import Gptq
from olive.passes.pytorch.kquant import KQuant, kquant_find_qparams
from olive.passes.pytorch.selective_mixed_precision import SelectiveMixedPrecision
from test.passes.pytorch.quantization_test_utils import (
    assert_packed_quant_module,
    load_quant_tensor_from_disk,
    make_local_calibration_data_config,
    make_local_tiny_dense_llama,
)

GROUP_SIZE = 16


def _assert_saved_quant_tensor_matches(
    output_path: Path,
    module_name: str,
    quant_tensor: QuantTensor,
) -> QuantTensor:
    disk_tensor = load_quant_tensor_from_disk(
        output_path,
        f"{module_name}.weight",
        bits=quant_tensor.bits,
        symmetric=quant_tensor.symmetric,
        group_size=quant_tensor.group_size,
        shape=tuple(quant_tensor.shape),
    )
    assert torch.equal(quant_tensor.qweight, disk_tensor.qweight)
    assert torch.equal(quant_tensor.scales, disk_tensor.scales)
    if quant_tensor.qzeros is None:
        assert disk_tensor.qzeros is None
    else:
        assert torch.equal(quant_tensor.qzeros, disk_tensor.qzeros)
    torch.testing.assert_close(quant_tensor.to_dense(), disk_tensor.to_dense(), rtol=0, atol=0)
    return disk_tensor


def _assert_uniform_int2_checkpoint(loaded, output_path: Path) -> None:
    quantized_linears = {
        f"model.layers.0.{name}": module
        for name, module in loaded.model.layers[0].named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    assert quantized_linears
    for module_name, module in quantized_linears.items():
        quant_tensor = assert_packed_quant_module(
            module,
            bits=2,
            group_size=GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name) == {
            "bits": 2,
            "symmetric": True,
            "group_size": GROUP_SIZE,
        }
        _assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)


def test_gptq_int2_dense_checkpoint_packing_and_numerics(tmp_path: Path):
    """GPTQ INT2 uses varied local calibration and preserves its packed checkpoint exactly."""
    input_model = make_local_tiny_dense_llama(tmp_path / "input_model")
    original_o_proj = input_model.load_model().model.layers[0].self_attn.o_proj.weight.detach().clone()
    quantizer = create_pass_from_dict(
        Gptq,
        {
            "bits": 2,
            "group_size": GROUP_SIZE,
            "sym": True,
            # Act ordering is only supported with group_size=-1. This grouped reference
            # configuration therefore disables it explicitly rather than relying on inference.
            "desc_act": False,
            "data_config": make_local_calibration_data_config(),
        },
        disable_search=True,
    )
    output_path = tmp_path / "gptq_int2"

    loaded = quantizer.run(input_model, str(output_path)).load_model().eval()

    _assert_uniform_int2_checkpoint(loaded, output_path)
    o_proj = loaded.model.layers[0].self_attn.o_proj
    disk_o_proj = _assert_saved_quant_tensor_matches(
        output_path,
        "model.layers.0.self_attn.o_proj",
        o_proj._parameters["weight"],  # pylint: disable=protected-access
    )

    # GPTQ's Hessian correction can move a weight by more than its final scale, so an
    # RTN-style max-scale tolerance is mathematically invalid. This assertion guards
    # against a degenerate serialized reconstruction; algorithm quality is covered by
    # dedicated GPTQ tests.
    reconstruction_mse = (disk_o_proj.to_dense() - original_o_proj).square().mean()
    zero_reconstruction_mse = original_o_proj.square().mean()
    assert reconstruction_mse < zero_reconstruction_mse

    inputs = torch.randn(2, o_proj.in_features, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        expected = torch.nn.functional.linear(inputs, disk_o_proj.to_dense(), o_proj.bias)
        actual = o_proj(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_kquant_int2_dense_checkpoint_matches_reference(tmp_path: Path):
    """KQuant INT2 preserves the quantizer result exactly after save/reload."""
    input_model = make_local_tiny_dense_llama(tmp_path / "input_model")
    original_o_proj = input_model.load_model().model.layers[0].self_attn.o_proj.weight.detach().clone()
    quantizer = create_pass_from_dict(
        KQuant,
        {"bits": 2, "group_size": GROUP_SIZE, "sym": True},
        disable_search=True,
    )
    output_path = tmp_path / "kquant_int2"

    loaded = quantizer.run(input_model, str(output_path)).load_model()

    _assert_uniform_int2_checkpoint(loaded, output_path)
    actual = loaded.model.layers[0].self_attn.o_proj._parameters["weight"]  # pylint: disable=protected-access

    # Recompute the KQuant result directly from the pre-algorithm float weight and
    # require the pass plumbing and serialized checkpoint to preserve it bit-exactly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    reference_weight = original_o_proj.to(device)
    maxq, minq = get_maxq_minq(2, signed=False)
    scales, zero_points = kquant_find_qparams(
        reference_weight,
        group_size=GROUP_SIZE,
        maxq=maxq,
        minq=minq,
        symmetric=True,
    )
    reference = QuantTensor.from_float(
        reference_weight,
        bits=2,
        symmetric=True,
        group_size=GROUP_SIZE,
        scales=scales,
        zero_points=zero_points,
    )
    assert torch.equal(actual.qweight, reference.qweight.cpu())
    assert torch.equal(actual.scales, reference.scales.cpu())
    torch.testing.assert_close(actual.to_dense(), reference.to_dense().cpu(), rtol=0, atol=0)

    # Also make the comparison to the original explicit: KQuant's selected
    # reconstruction must outperform the all-zero baseline.
    assert (actual.to_dense() - original_o_proj).square().mean() < original_o_proj.square().mean()


@pytest.mark.parametrize(
    ("pass_class", "extra_config"),
    [
        pytest.param(
            Gptq,
            {"desc_act": False, "data_config": make_local_calibration_data_config()},
            id="gptq",
        ),
        pytest.param(KQuant, {}, id="kquant"),
    ],
)
def test_selective_mixed_precision_int2_int4_int8_consumed_by_dense_quantizer(
    tmp_path: Path,
    pass_class,
    extra_config: dict,
):
    """SMP INT2 defaults, selected INT4, and explicit INT8 overrides survive save/reload."""
    input_model = make_local_tiny_dense_llama(tmp_path / "input_model")
    planner = create_pass_from_dict(
        SelectiveMixedPrecision,
        {"algorithm": "high_precision_mlp_down", "bits": 2, "high_bits": 4},
        disable_search=True,
    )
    planned = planner.run(input_model, str(tmp_path / "smp"))
    assert planned.model_attributes["mixed_precision_info"]["default"] == {"bits": 2}
    assert planned.model_attributes["mixed_precision_info"]["overrides"]["model.layers.0.mlp.down_proj"] == {"bits": 4}

    quantizer = create_pass_from_dict(
        pass_class,
        {
            "group_size": GROUP_SIZE,
            "sym": True,
            "overrides": {"model.layers.0.mlp.gate_proj": {"bits": 8}},
            **extra_config,
        },
        disable_search=True,
    )
    output_path = tmp_path / pass_class.__name__.lower()
    loaded = quantizer.run(planned, str(output_path)).load_model()

    expected_bits = {
        "model.layers.0.mlp.up_proj": 2,
        "model.layers.0.mlp.down_proj": 4,
        "model.layers.0.mlp.gate_proj": 8,
    }
    for module_name, bits in expected_bits.items():
        quant_tensor = assert_packed_quant_module(
            loaded.get_submodule(module_name),
            bits=bits,
            group_size=GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name) == {
            "bits": bits,
            "symmetric": True,
            "group_size": GROUP_SIZE,
        }
        _assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)

    # Keep the established QKV grouping behavior unchanged: absent a QKV-specific
    # override, all three projections consume the SMP INT2 default.
    for projection in ("q_proj", "k_proj", "v_proj"):
        module_name = f"model.layers.0.self_attn.{projection}"
        quant_tensor = assert_packed_quant_module(
            loaded.get_submodule(module_name),
            bits=2,
            group_size=GROUP_SIZE,
            symmetric=True,
        )
        assert loaded.config.quantization_config.get_qlinear_init_args(module_name)["bits"] == 2
        _assert_saved_quant_tensor_matches(output_path, module_name, quant_tensor)
