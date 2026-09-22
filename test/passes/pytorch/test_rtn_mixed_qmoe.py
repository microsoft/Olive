# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from pathlib import Path

import pytest
import torch

from olive.common.quant.hf_utils import OliveHfQuantizationConfig
from olive.common.quant.selection import iter_quant_targets
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import unpack_from_uint8
from olive.model import HfModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.pytorch.rtn import Rtn

_NUM_LAYERS = 2
_NUM_EXPERTS = 3
_HIDDEN_SIZE = 32
_MOE_INTERMEDIATE_SIZE = 48
_GROUP_SIZE = 16


def _make_local_tiny_qwen3_moe(save_path: Path):
    """Create a deterministic fused K-last Qwen3-MoE checkpoint without hub access."""
    from transformers import Qwen3MoeConfig, Qwen3MoeForCausalLM

    torch.manual_seed(0)
    config = Qwen3MoeConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=32,
        moe_intermediate_size=_MOE_INTERMEDIATE_SIZE,
        num_hidden_layers=_NUM_LAYERS,
        num_attention_heads=4,
        num_key_value_heads=4,
        num_experts=_NUM_EXPERTS,
        num_experts_per_tok=1,
        decoder_sparse_step=1,
        head_dim=8,
    )
    model = Qwen3MoeForCausalLM(config).eval()
    model.set_experts_implementation("eager")
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, save_original_format=False)


def _make_local_tiny_qwen3_5_moe_vl(save_path: Path):
    """Create a tiny nested Qwen3.5-MoE VL checkpoint without hub access."""
    from transformers import (
        Qwen3_5MoeConfig,
        Qwen3_5MoeForConditionalGeneration,
        Qwen3_5MoeTextConfig,
        Qwen3_5MoeVisionConfig,
    )

    torch.manual_seed(0)
    text_config = Qwen3_5MoeTextConfig(  # pylint: disable=unexpected-keyword-arg
        vocab_size=32,
        hidden_size=_HIDDEN_SIZE,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        moe_intermediate_size=_MOE_INTERMEDIATE_SIZE,
        shared_expert_intermediate_size=32,
        num_experts_per_tok=1,
        num_experts=_NUM_EXPERTS,
        layer_types=["full_attention"],
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_num_key_heads=4,
        linear_num_value_heads=4,
    )
    vision_config = Qwen3_5MoeVisionConfig(  # pylint: disable=unexpected-keyword-arg
        depth=1,
        hidden_size=_HIDDEN_SIZE,
        intermediate_size=64,
        num_heads=4,
        out_hidden_size=_HIDDEN_SIZE,
        num_position_embeddings=16,
        patch_size=2,
        spatial_merge_size=1,
        temporal_patch_size=1,
    )
    config = Qwen3_5MoeConfig(  # pylint: disable=unexpected-keyword-arg
        text_config=text_config,
        vision_config=vision_config,
        image_token_id=29,
        video_token_id=30,
        vision_start_token_id=27,
        vision_end_token_id=28,
    )
    model = Qwen3_5MoeForConditionalGeneration(config).eval()
    save_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_path, save_original_format=False)


def _mixed_qmoe_overrides(kind: str) -> dict:
    if kind == "exact":
        return {
            **{f"model.layers.{layer}.mlp.experts.gate_up_proj": {"bits": 2} for layer in range(_NUM_LAYERS)},
            **{f"model.layers.{layer}.mlp.experts.down_proj": {"bits": 4} for layer in range(_NUM_LAYERS)},
        }
    if kind == "regex":
        return {
            r"re:model\.layers\.\d+\.mlp\.experts\.gate_up_proj": {"bits": 2},
            r"re:model\.layers\.\d+\.mlp\.experts\.down_proj": {"bits": 4},
        }
    raise ValueError(f"Unknown override kind: {kind}")


def _mixed_qmoe_regex_overrides(source_prefix: str) -> dict:
    escaped_prefix = source_prefix.replace(".", r"\.")
    return {
        rf"re:{escaped_prefix}\.\d+\.mlp\.experts\.gate_up_proj": {"bits": 2},
        rf"re:{escaped_prefix}\.\d+\.mlp\.experts\.down_proj": {"bits": 4},
    }


def _target_tensor(model, name: str):
    try:
        return model.get_parameter(name).data
    except AttributeError:
        return model.get_submodule(name).weight.data


def _quant_tensor_snapshot(model, target_names: list[str]) -> dict[str, tuple]:
    snapshot = {}
    for name in target_names:
        quant_tensor = _target_tensor(model, name)
        assert isinstance(quant_tensor, QuantTensor), f"{name} silently remained in floating point"
        snapshot[name] = (
            quant_tensor.bits,
            quant_tensor.group_size,
            quant_tensor.symmetric,
            quant_tensor.qweight.clone(),
            quant_tensor.scales.clone(),
            None if quant_tensor.qzeros is None else quant_tensor.qzeros.clone(),
        )
    return snapshot


def _assert_snapshots_equal(actual: dict[str, tuple], expected: dict[str, tuple]) -> None:
    assert actual.keys() == expected.keys()
    for name, actual_values in actual.items():
        actual_bits, actual_group_size, actual_sym, actual_qweight, actual_scales, actual_qzeros = actual_values
        expected_bits, expected_group_size, expected_sym, expected_qweight, expected_scales, expected_qzeros = expected[
            name
        ]
        assert (actual_bits, actual_group_size, actual_sym) == (
            expected_bits,
            expected_group_size,
            expected_sym,
        )
        torch.testing.assert_close(actual_qweight, expected_qweight, rtol=0, atol=0)
        torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
        if expected_qzeros is None:
            assert actual_qzeros is None
        else:
            torch.testing.assert_close(actual_qzeros, expected_qzeros, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("source_prefix", "module_prefix"),
    [
        pytest.param(
            "model.layers",
            "model.layers.0.mlp.experts",
            id="qwen3-or-qwen3_5-text",
        ),
        pytest.param(
            "model.language_model.layers",
            "model.language_model.layers.0.mlp.experts",
            id="qwen3_5-vl",
        ),
    ],
)
def test_mixed_qmoe_regex_overrides_resolve_source_prefix(source_prefix: str, module_prefix: str):
    qcfg = OliveHfQuantizationConfig(
        bits=4,
        symmetric=False,
        group_size=_GROUP_SIZE,
        moe=True,
        overrides=_mixed_qmoe_regex_overrides(source_prefix),
    )

    assert qcfg.get_qlinear_init_args(f"{module_prefix}.gate_up_proj")["bits"] == 2
    assert qcfg.get_qlinear_init_args(f"{module_prefix}.down_proj")["bits"] == 4


def test_flat_text_regex_does_not_match_qwen3_5_vl_source_prefix():
    qcfg = OliveHfQuantizationConfig(
        bits=4,
        symmetric=False,
        group_size=_GROUP_SIZE,
        moe=True,
        overrides=_mixed_qmoe_regex_overrides("model.layers"),
    )

    qwen3_5_vl_fc1 = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    assert qcfg.get_qlinear_init_args(qwen3_5_vl_fc1)["bits"] == 4


def test_rtn_mixed_qmoe_qwen3_5_vl_nested_layout_roundtrip(tmp_path: Path):
    """Run mixed-width RTN through the nested Qwen3.5-VL text backbone."""
    source_path = tmp_path / "source"
    _make_local_tiny_qwen3_5_moe_vl(source_path)
    expert_names = {
        f"model.language_model.layers.0.mlp.experts.{projection}" for projection in ("gate_up_proj", "down_proj")
    }
    source_model = HfModelHandler(model_path=str(source_path), task="image-text-to-text")
    float_model = source_model.load_model().eval()
    target_names = {
        name
        for _, _, name in iter_quant_targets(
            float_model,
            quantize_lm_head=False,
            quantize_embeds=False,
            quantize_moe=True,
        )
    }
    assert expert_names.issubset(target_names)
    rtn = create_pass_from_dict(
        Rtn,
        {
            "bits": 4,
            "group_size": _GROUP_SIZE,
            "sym": False,
            "moe": True,
            "overrides": _mixed_qmoe_regex_overrides("model.language_model.layers"),
        },
        disable_search=True,
    )

    output = rtn.run(source_model, str(tmp_path / "rtn"))
    from safetensors import safe_open

    gate = "model.language_model.layers.0.mlp.experts.gate_up_proj"
    down = "model.language_model.layers.0.mlp.experts.down_proj"
    expected_payload = {
        gate + "_qweight": (
            (_NUM_EXPERTS, 2 * _MOE_INTERMEDIATE_SIZE, _HIDDEN_SIZE * 2 // 8),
            torch.uint8,
        ),
        gate + "_scales": (
            (
                _NUM_EXPERTS,
                2 * _MOE_INTERMEDIATE_SIZE,
                _HIDDEN_SIZE // _GROUP_SIZE,
            ),
            torch.float32,
        ),
        gate + "_qzeros": (
            (
                _NUM_EXPERTS,
                2 * _MOE_INTERMEDIATE_SIZE,
                ((_HIDDEN_SIZE // _GROUP_SIZE) * 2 + 7) // 8,
            ),
            torch.uint8,
        ),
        down + "_qweight": (
            (_NUM_EXPERTS, _HIDDEN_SIZE, _MOE_INTERMEDIATE_SIZE * 4 // 8),
            torch.uint8,
        ),
        down + "_scales": (
            (
                _NUM_EXPERTS,
                _HIDDEN_SIZE,
                _MOE_INTERMEDIATE_SIZE // _GROUP_SIZE,
            ),
            torch.float32,
        ),
        down + "_qzeros": (
            (
                _NUM_EXPERTS,
                _HIDDEN_SIZE,
                ((_MOE_INTERMEDIATE_SIZE // _GROUP_SIZE) * 4 + 7) // 8,
            ),
            torch.uint8,
        ),
    }
    serialized_payload = {}
    for shard in Path(output.model_path).glob("*.safetensors"):
        with safe_open(shard, framework="pt") as handle:
            tensor_names = handle.keys()
            for name in tensor_names:
                if name.startswith("model.language_model.layers.0.mlp.experts."):
                    serialized_payload[name] = handle.get_tensor(name)

    assert set(serialized_payload) == set(expected_payload)
    for name, (shape, dtype) in expected_payload.items():
        assert serialized_payload[name].shape == shape
        assert serialized_payload[name].dtype == dtype
    assert sum(tensor.numel() * tensor.element_size() for tensor in serialized_payload.values()) == sum(
        math.prod(shape) * torch.empty((), dtype=dtype).element_size() for shape, dtype in expected_payload.values()
    )

    quantized_model = output.load_model().eval()
    first_snapshot = _quant_tensor_snapshot(quantized_model, sorted(expert_names))
    assert first_snapshot[next(name for name in expert_names if name.endswith("gate_up_proj"))][0] == 2
    assert first_snapshot[next(name for name in expert_names if name.endswith("down_proj"))][0] == 4

    reload_path = tmp_path / "reload"
    quantized_model.save_pretrained(reload_path, save_original_format=False)
    reloaded_model = (
        HfModelHandler(
            model_path=str(reload_path),
            task="image-text-to-text",
        )
        .load_model()
        .eval()
    )
    _assert_snapshots_equal(
        _quant_tensor_snapshot(reloaded_model, sorted(expert_names)),
        first_snapshot,
    )


@pytest.mark.parametrize("symmetric", [True, False])
def test_rtn_mixed_qmoe_exact_and_regex_overrides_roundtrip(tmp_path: Path, symmetric: bool):
    """Qualify deterministic fused FC1 INT2 / FC2 INT4 RTN checkpoints."""
    source_path = tmp_path / "source"
    _make_local_tiny_qwen3_moe(source_path)
    float_model = HfModelHandler(model_path=str(source_path)).load_model().eval()
    input_ids = torch.tensor([[1, 2, 3, 4]])

    target_names = [
        name
        for _, _, name in iter_quant_targets(
            float_model,
            quantize_lm_head=False,
            quantize_embeds=False,
            quantize_moe=True,
        )
    ]
    expert_names = {
        f"model.layers.{layer}.mlp.experts.{projection}"
        for layer in range(_NUM_LAYERS)
        for projection in ("gate_up_proj", "down_proj")
    }
    assert expert_names.issubset(target_names)
    snapshots = {}
    effective_assignments = {}
    quantized_logits = {}
    for override_kind in ("exact", "regex"):
        rtn = create_pass_from_dict(
            Rtn,
            {
                "bits": 4,
                "group_size": _GROUP_SIZE,
                "sym": symmetric,
                "moe": True,
                "overrides": _mixed_qmoe_overrides(override_kind),
            },
            disable_search=True,
        )
        output = rtn.run(HfModelHandler(model_path=str(source_path)), str(tmp_path / f"rtn_{override_kind}"))
        quantized_model = output.load_model().eval()
        quantized_model.set_experts_implementation("eager")

        qcfg = quantized_model.config.quantization_config
        effective_assignments[override_kind] = {name: qcfg.get_qlinear_init_args(name) for name in target_names}
        for name, assignment in effective_assignments[override_kind].items():
            assert assignment == {
                "bits": 2 if name.endswith("experts.gate_up_proj") else 4,
                "symmetric": symmetric,
                "group_size": _GROUP_SIZE,
            }

        # Explicit FC2=INT4 overrides may be retained or elided during
        # serialization; only the effective assignment is contractual.

        first_snapshot = _quant_tensor_snapshot(quantized_model, target_names)
        for name in expert_names:
            quant_tensor = _target_tensor(quantized_model, name)
            unpacked_qweight = unpack_from_uint8(
                quant_tensor.qweight,
                quant_tensor.bits,
                tuple(quant_tensor.shape),
            )
            assert unpacked_qweight.shape == quant_tensor.shape
            assert unpacked_qweight.min() >= 0
            assert unpacked_qweight.max() <= (1 << quant_tensor.bits) - 1
            assert torch.isfinite(quant_tensor.scales).all()
            if symmetric:
                assert quant_tensor.qzeros is None
            else:
                unpacked_qzeros = unpack_from_uint8(
                    quant_tensor.qzeros,
                    quant_tensor.bits,
                    tuple(quant_tensor.scales.shape),
                )
                assert unpacked_qzeros.shape == quant_tensor.scales.shape
                assert unpacked_qzeros.min() >= 0
                assert unpacked_qzeros.max() <= (1 << quant_tensor.bits) - 1

        for layer in range(_NUM_LAYERS):
            experts = quantized_model.model.layers[layer].mlp.experts
            gate_up = experts.gate_up_proj.data
            down = experts.down_proj.data
            assert gate_up.bits == 2
            assert gate_up.qweight.shape == (_NUM_EXPERTS, 2 * _MOE_INTERMEDIATE_SIZE, _HIDDEN_SIZE // 4)
            assert gate_up.scales.shape == (
                _NUM_EXPERTS,
                2 * _MOE_INTERMEDIATE_SIZE,
                _HIDDEN_SIZE // _GROUP_SIZE,
            )
            assert down.bits == 4
            assert down.qweight.shape == (_NUM_EXPERTS, _HIDDEN_SIZE, _MOE_INTERMEDIATE_SIZE // 2)
            assert down.scales.shape == (
                _NUM_EXPERTS,
                _HIDDEN_SIZE,
                _MOE_INTERMEDIATE_SIZE // _GROUP_SIZE,
            )
            if symmetric:
                assert gate_up.qzeros is None
                assert down.qzeros is None
            else:
                assert gate_up.qzeros.shape == (_NUM_EXPERTS, 2 * _MOE_INTERMEDIATE_SIZE, 1)
                # Three INT4 zero points require ceil(3 / 2) == 2 packed bytes.
                assert down.qzeros.shape == (_NUM_EXPERTS, _HIDDEN_SIZE, 2)

        with torch.no_grad():
            before_reload_logits = quantized_model(input_ids).logits
        assert torch.isfinite(before_reload_logits).all()
        reload_path = tmp_path / f"reload_{override_kind}"
        quantized_model.save_pretrained(reload_path, save_original_format=False)
        reloaded_model = HfModelHandler(model_path=str(reload_path)).load_model().eval()
        reloaded_model.set_experts_implementation("eager")
        reloaded_qcfg = reloaded_model.config.quantization_config
        assert {name: reloaded_qcfg.get_qlinear_init_args(name) for name in target_names} == effective_assignments[
            override_kind
        ]
        after_reload_snapshot = _quant_tensor_snapshot(reloaded_model, target_names)
        _assert_snapshots_equal(after_reload_snapshot, first_snapshot)
        with torch.no_grad():
            after_reload_logits = reloaded_model(input_ids).logits
        assert torch.isfinite(after_reload_logits).all()
        torch.testing.assert_close(after_reload_logits, before_reload_logits, rtol=0, atol=0)

        snapshots[override_kind] = after_reload_snapshot
        quantized_logits[override_kind] = after_reload_logits

    assert effective_assignments["exact"] == effective_assignments["regex"]
    _assert_snapshots_equal(snapshots["exact"], snapshots["regex"])
    torch.testing.assert_close(quantized_logits["exact"], quantized_logits["regex"], rtol=0, atol=0)
