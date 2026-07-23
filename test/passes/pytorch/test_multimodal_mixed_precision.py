# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the MultiModalMixedPrecision pass and its component classifier."""

# Tests call the pure planner helper ``MultiModalMixedPrecision._build_plan`` directly.
# pylint: disable=protected-access
import pytest

from olive.passes.pytorch.multimodal_mixed_precision import (
    Component,
    MultiModalMixedPrecision,
    classify_component,
)

# Representative module names spanning Qwen-VL, Gemma-4 (omni), and Phi-style VLMs.
_QWEN3VL = [
    "model.language_model.layers.0.self_attn.q_proj",
    "model.language_model.layers.3.mlp.down_proj",
    "model.visual.blocks.0.attn.qkv",
    "model.visual.blocks.10.mlp.fc2",
    "model.visual.merger.linear_fc1",
]
_GEMMA4 = [
    "model.language_model.layers.1.mlp.up_proj",
    "model.vision_tower.encoder.layers.0.self_attn.k_proj",
    "model.audio_tower.layers.2.conformer.ffn.linear",
]


class TestClassifyComponent:
    @pytest.mark.parametrize(
        ("name", "expected"),
        [
            ("model.language_model.layers.0.self_attn.q_proj", Component.TEXT),
            ("model.language_model.layers.3.mlp.down_proj", Component.TEXT),
            ("model.visual.blocks.0.attn.qkv", Component.VISION),
            ("model.vision_tower.encoder.layers.0.self_attn.k_proj", Component.VISION),
            ("vision_model.embeddings.patch_embedding", Component.VISION),
            ("model.audio_tower.layers.2.conformer.ffn.linear", Component.AUDIO),
            ("model.audio_encoder.blocks.0.attn.out_proj", Component.AUDIO),
            ("model.multi_modal_projector.linear_1", Component.PROJECTOR),
            ("model.visual.merger.linear_fc1", Component.PROJECTOR),
        ],
    )
    def test_classify_component_matches_expected_component(self, name, expected):
        assert classify_component(name) == expected

    def test_classify_component_prefers_explicit_lm_head_and_embeds(self):
        assert classify_component("lm_head", lm_head_name="lm_head") == Component.LM_HEAD
        assert (
            classify_component("model.language_model.embed_tokens", embeds_name="model.language_model.embed_tokens")
            == Component.EMBEDS
        )
        gemma_embedding_modules = {
            "model.language_model.embed_tokens",
            "model.language_model.embed_tokens_per_layer",
            "model.language_model.per_layer_model_projection",
            "model.language_model.per_layer_projection_norm",
        }
        assert (
            classify_component(
                "model.language_model.per_layer_model_projection",
                embeds_names=gemma_embedding_modules,
            )
            == Component.EMBEDS
        )
        assert (
            classify_component(
                "model.language_model.layers.0.per_layer_projection",
                embeds_names=gemma_embedding_modules,
            )
            == Component.TEXT
        )

    def test_classify_component_defaults_to_text_when_unmatched(self):
        assert classify_component("model.language_model.layers.0.mlp.gate_proj") == Component.TEXT

    def test_classify_component_projector_wins_over_vision_when_nested(self):
        # merger lives under the vision tower but must classify as projector (rule order).
        assert classify_component("model.visual.merger.linear_fc2") == Component.PROJECTOR

    def test_classify_component_honors_extra_rules_first(self):
        # A custom tower name unknown to the built-in heuristics.
        assert classify_component("model.perception.block0.proj") == Component.TEXT
        assert (
            classify_component("model.perception.block0.proj", extra_rules=[(["perception"], "vision")])
            == Component.VISION
        )


class TestBuildPlan:
    @staticmethod
    def _modules(names):
        return [(n, "linear") for n in names]

    def test_build_plan_excludes_full_precision_components(self):
        exclude, overrides, counts = MultiModalMixedPrecision._build_plan(
            self._modules(_QWEN3VL),
            component_precision={"vision": 16, "text": 4},
            default_bits=4,
        )
        # All vision + projector-under-vision handled: vision excluded, text quantized at default.
        assert all("visual.blocks" in n for n in exclude)
        assert "model.visual.merger.linear_fc1" not in exclude  # projector, not vision
        assert not overrides  # text == default bits -> no override
        assert counts["vision"] == 2
        assert counts["text"] == 2
        assert counts["projector"] == 1

    def test_build_plan_emits_overrides_for_nondefault_quantized_component(self):
        exclude, overrides, _ = MultiModalMixedPrecision._build_plan(
            self._modules(_GEMMA4),
            component_precision={"vision": 16, "audio": 8, "text": 4},
            default_bits=4,
        )
        assert exclude == ["model.vision_tower.encoder.layers.0.self_attn.k_proj"]
        assert overrides == {"model.audio_tower.layers.2.conformer.ffn.linear": {"bits": 8}}

    def test_build_plan_no_op_when_component_precision_empty(self):
        exclude, overrides, _ = MultiModalMixedPrecision._build_plan(
            self._modules(_QWEN3VL), component_precision={}, default_bits=4
        )
        assert not exclude
        assert not overrides

    def test_build_plan_default_bits_component_not_overridden(self):
        # text set to the same as default -> should not appear in overrides
        _, overrides, _ = MultiModalMixedPrecision._build_plan(
            self._modules(_QWEN3VL), component_precision={"text": 4}, default_bits=4
        )
        assert not overrides

    def test_build_plan_raises_on_unsupported_bits(self):
        with pytest.raises(ValueError, match="Unsupported bits"):
            MultiModalMixedPrecision._build_plan(
                self._modules(_QWEN3VL), component_precision={"text": 5}, default_bits=4
            )
