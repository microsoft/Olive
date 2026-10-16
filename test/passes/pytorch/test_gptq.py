# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import pack_to_uint8
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch import quant_utils as quant_utils_module
from olive.passes.pytorch.gptq import Gptq
from olive.passes.pytorch.train_utils import get_calibration_dataset
from test.passes.pytorch.test_quantization_utils import (
    DENSE_INT2_GROUP_SIZE,
    assert_dense_int2_mixed_precision_checkpoint,
    assert_saved_quant_tensor_matches,
    assert_uniform_int2_checkpoint,
    make_local_calibration_data_config,
    make_local_tiny_dense_llama,
    make_local_tiny_tied_gemma4,
    plan_dense_int2_mixed_precision,
)
from test.utils import get_tiny_phi3, make_local_tiny_llama


def _is_quant(module: torch.nn.Module) -> bool:
    if not isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
        return False
    weight = module._parameters.get("weight")
    return weight is not None and isinstance(weight.data, QuantTensor)


def _bits(module: torch.nn.Module) -> int:
    return module.weight.data.bits


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_gptq_rejects_non_finite_moe_fallback_min_k_multiple(value):
    p = create_pass_from_dict(
        Gptq,
        {"moe_fallback_min_k_multiple": value},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider"),
    )

    assert not Gptq.validate_config(p.config, p.accelerator_spec)


def test_gptq_int2_dense_checkpoint_packing_and_numerics(tmp_path: Path):
    """GPTQ INT2 uses varied local calibration and preserves its packed checkpoint exactly."""
    input_model = make_local_tiny_dense_llama(tmp_path / "input_model")
    original_o_proj = input_model.load_model().model.layers[0].self_attn.o_proj.weight.detach().clone()
    quantizer = create_pass_from_dict(
        Gptq,
        {
            "bits": 2,
            "group_size": DENSE_INT2_GROUP_SIZE,
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

    assert_uniform_int2_checkpoint(loaded, output_path)
    o_proj = loaded.model.layers[0].self_attn.o_proj
    disk_o_proj = assert_saved_quant_tensor_matches(
        output_path,
        "model.layers.0.self_attn.o_proj",
        o_proj._parameters["weight"],
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
        expected = inputs @ disk_o_proj.to_dense().T
        if o_proj.bias is not None:
            expected += o_proj.bias
        actual = o_proj(inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_gptq_consumes_selective_mixed_precision_int2_int4_int8(tmp_path: Path):
    """GPTQ materializes SMP INT2 defaults, selected INT4, and an explicit INT8 override."""
    input_model = make_local_tiny_dense_llama(tmp_path / "input_model")
    planned = plan_dense_int2_mixed_precision(input_model, tmp_path / "smp")
    quantizer = create_pass_from_dict(
        Gptq,
        {
            "group_size": DENSE_INT2_GROUP_SIZE,
            "sym": True,
            "overrides": {"model.layers.0.mlp.gate_proj": {"bits": 8}},
            "desc_act": False,
            "data_config": make_local_calibration_data_config(),
        },
        disable_search=True,
    )
    output_path = tmp_path / "gptq_mixed"

    loaded = quantizer.run(planned, str(output_path)).load_model()

    assert_dense_int2_mixed_precision_checkpoint(loaded, output_path)


@pytest.mark.parametrize("group_size", [16, -1])
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("bits", [4, 8])
def test_gptq_embedding_calibration_checkpoint_and_eager_output(tmp_path, monkeypatch, group_size, symmetric, bits):
    pytest.importorskip("transformers.models.gemma4")
    input_model = make_local_tiny_tied_gemma4(tmp_path / "input")
    input_model.model_attributes = {
        "component_name": "embedding",
        "component_role": "embedding",
        "component_source_paths": [
            "model.language_model.embed_tokens",
            "model.language_model.embed_tokens_per_layer",
            "model.language_model.per_layer_model_projection",
            "model.language_model.per_layer_projection_norm",
        ],
    }
    data_config = make_local_calibration_data_config(seq_len=8, max_samples=4)
    token_ids = torch.cat(
        [sample["input_ids"].flatten() for sample in get_calibration_dataset(input_model, data_config)]
    )
    expected_counts = torch.bincount(token_ids, minlength=128).float()
    quantizer = create_pass_from_dict(
        Gptq,
        {
            "bits": bits,
            "group_size": group_size,
            "sym": symmetric,
            "desc_act": False,
            "data_config": data_config,
        },
        disable_search=True,
    )
    monkeypatch.setattr(quant_utils_module, "_resolve_layerwise_device", lambda _: "cpu")
    recorded = {}
    process_module = quantizer.process_module

    def record_embedding(module, **kwargs):
        if isinstance(module, torch.nn.Embedding):
            data = module.weight.quant_info.data
            assert data["H"].shape == (128,)
            assert data["N"] == token_ids.numel()
            torch.testing.assert_close(data["H"], expected_counts, rtol=0, atol=0)
        process_module(module, **kwargs)
        if isinstance(module, torch.nn.Embedding):
            info = module.weight.quant_info
            assert info.data is None
            recorded[module.embedding_dim] = (
                module.weight.detach().clone(),
                info.scales.clone(),
                info.zero_points.clone(),
            )

    monkeypatch.setattr(quantizer, "process_module", record_embedding)
    output_path = tmp_path / "quantized"
    loaded = quantizer.run(input_model, str(output_path)).load_model().eval()
    assert set(recorded) == {64, 32}
    assert loaded.config.quantization_config.embeds is True
    assert loaded.config.quantization_config.lm_head is False
    backbone = loaded.model.language_model
    assert not _is_quant(loaded.lm_head)
    assert not _is_quant(backbone.layers[0].self_attn.q_proj)
    ids = torch.tensor([[1, 2, 3, 4]])
    for name in ("embed_tokens", "embed_tokens_per_layer"):
        module = getattr(backbone, name)
        assert _is_quant(module)
        quant_tensor = module.weight.data
        assert quant_tensor.bits == bits
        assert quant_tensor.group_size == group_size
        assert quant_tensor.symmetric is symmetric
        expected_weight, expected_scales, expected_zero_points = recorded[module.embedding_dim]
        groups = module.embedding_dim // group_size if group_size > 0 else 1
        assert quant_tensor.scales.shape == (128, groups)
        assert torch.equal(quant_tensor.scales, expected_scales)
        if symmetric:
            assert quant_tensor.qzeros is None
        else:
            assert torch.equal(quant_tensor.qzeros, pack_to_uint8(expected_zero_points, bits))
        disk_tensor = assert_saved_quant_tensor_matches(output_path, f"model.language_model.{name}", quant_tensor)
        torch.testing.assert_close(disk_tensor.to_dense(), expected_weight)
        with torch.no_grad():
            expected = torch.nn.functional.embedding(ids, disk_tensor.to_dense()) * module.embed_scale
            torch.testing.assert_close(module(ids), expected)


def test_gptq_embedding_component_rejects_effective_lm_head(tmp_path):
    pytest.importorskip("transformers.models.gemma4")
    input_model = make_local_tiny_tied_gemma4(tmp_path / "input")
    input_model.model_attributes = {
        "component_name": "embedding",
        "component_role": "embedding",
        "component_source_paths": ["model.language_model.embed_tokens"],
        "mixed_precision_info": {
            "default": {"lm_head": True},
            "overrides": {},
        },
    }
    quantizer = create_pass_from_dict(Gptq, {}, disable_search=True)

    with pytest.raises(ValueError, match="embedding component does not own lm_head"):
        quantizer.run(input_model, str(tmp_path / "quantized"))


# running on CPU takes time so will only run a subset of tests when GPU is not available
@pytest.mark.parametrize(
    ("model_path", "expected_model_type"),
    [
        ("tiny-phi3", "Phi3ForCausalLM"),
        ("tiny-llama", "LlamaForCausalLM"),
    ],
)
@pytest.mark.parametrize("group_size", [-1, 16] if torch.cuda.is_available() else [16])
@pytest.mark.parametrize("lm_head", [True, False] if torch.cuda.is_available() else [False])
@pytest.mark.parametrize("sym", [True, False] if torch.cuda.is_available() else [False])
def test_gptq(tmp_path: Path, model_path: str, expected_model_type: str, group_size: int, lm_head: bool, sym: bool):
    # setup
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "input_model")
    else:
        input_model = get_tiny_phi3()
    p = create_pass_from_dict(
        Gptq,
        {
            "group_size": group_size,
            "lm_head": lm_head,
            "sym": sym,
            "overrides": {"model.layers.0.self_attn.o_proj": {"bits": 8}},
        },
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="CUDAExecutionProvider"),
    )
    gptq_out_folder = str(tmp_path / "gptq")

    # execute
    out = p.run(input_model, gptq_out_folder)

    # assert
    assert isinstance(out, HfModelHandler)
    loaded_model = out.load_model()
    assert loaded_model.__class__.__name__ == expected_model_type
    assert hasattr(loaded_model, "quantization_method")
    assert loaded_model.quantization_method == "olive"
    assert hasattr(loaded_model.config, "quantization_config")
    assert isinstance(loaded_model.config.quantization_config, OliveHfQuantizationConfig)
    assert loaded_model.config.quantization_config.group_size == group_size
    assert not any(isinstance(m, torch.nn.Linear) and not _is_quant(m) for m in loaded_model.model.layers.modules())
    assert _is_quant(loaded_model.model.layers[0].self_attn.o_proj)
    assert _bits(loaded_model.model.layers[0].self_attn.o_proj) == 8
    assert _bits(loaded_model.model.layers[0].mlp.down_proj) == 4
    assert loaded_model.config.quantization_config.lm_head == lm_head
    assert _is_quant(loaded_model.lm_head) == lm_head
