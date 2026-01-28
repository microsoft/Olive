# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.io_resolver import get_task_template, resolve_alias
from olive.common.hf.io_config.task_config import (
    _build_inputs,
    generate_dummy_inputs,
    get_diffusers_io_config,
    get_io_config,
)
from olive.common.hf.io_config.tasks import TaskType


class TestResolveAlias:
    def test_num_hidden_layers(self):
        class Config:
            num_hidden_layers = 12

        assert resolve_alias(Config(), "num_layers") == 12

    def test_n_layer_fallback(self):
        class Config:
            n_layer = 24

        assert resolve_alias(Config(), "num_layers") == 24

    def test_missing_returns_none(self):
        class Config:
            pass

        assert resolve_alias(Config(), "num_layers") is None


class TestGetIOConfig:
    def test_text_classification(self):
        config = MagicMock()
        config.num_hidden_layers = 12

        io_config = get_io_config(config, TaskType.TEXT_CLASSIFICATION)

        assert "input_ids" in io_config["input_names"]
        assert "logits" in io_config["output_names"]
        assert io_config["dynamic_axes"]["input_ids"] == {0: "batch_size", 1: "sequence_length"}

    def test_text_generation_with_past(self):
        config = MagicMock()
        config.num_hidden_layers = 12

        io_config = get_io_config(config, TaskType.TEXT_GENERATION, use_past=True)

        # use_past flag is accepted but doesn't add flattened present outputs
        # (ONNX exporter infers output structure from traced model)
        assert "logits" in io_config["output_names"]

    def test_text_generation_with_past_in_inputs(self):
        config = MagicMock()
        config.num_hidden_layers = 12

        io_config = get_io_config(config, TaskType.TEXT_GENERATION, use_past=True, use_past_in_inputs=True)

        # input_names should have flattened past_key_values entries
        # This is required for torch.onnx.export to properly name the inputs
        assert "past_key_values.0.key" in io_config["input_names"]
        assert "past_key_values.0.value" in io_config["input_names"]
        assert "past_key_values.11.key" in io_config["input_names"]
        assert "past_key_values.11.value" in io_config["input_names"]

        # dynamic_axes should have flattened names for downstream processing (quantization, etc.)
        assert io_config["dynamic_axes"]["attention_mask"] == {
            0: "batch_size",
            1: "past_sequence_length + sequence_length",
        }
        assert "past_key_values.0.key" in io_config["dynamic_axes"]
        assert "past_key_values.0.value" in io_config["dynamic_axes"]
        assert "present.0.key" in io_config["dynamic_axes"]
        assert "present.0.value" in io_config["dynamic_axes"]

        # Verify all layers are present (num_hidden_layers = 12)
        assert "past_key_values.11.key" in io_config["dynamic_axes"]
        assert "present.11.value" in io_config["dynamic_axes"]

    def test_with_model_filters_optional_inputs(self):
        config = MagicMock()
        config.num_hidden_layers = 12

        # Model without token_type_ids in signature
        model = MagicMock()
        model.forward = lambda self, input_ids, attention_mask: None

        io_config = get_io_config(config, TaskType.TEXT_CLASSIFICATION, model=model)

        # token_type_ids is optional and not in model signature
        assert "token_type_ids" not in io_config["input_names"]

    def test_unsupported_task_raises(self):
        config = MagicMock()
        with pytest.raises(ValueError, match="Unsupported task"):
            get_io_config(config, "unsupported-task")


class TestBuildInputs:
    def test_filters_optional_by_model_signature(self):
        task_template = get_task_template(TaskType.TEXT_CLASSIFICATION)

        # Model with only input_ids
        model = MagicMock()
        model.forward = lambda self, input_ids: None

        inputs = _build_inputs(task_template, model)

        assert "input_ids" in inputs
        # Optional inputs not in model signature should be filtered
        assert "token_type_ids" not in inputs


class TestGenerateDummyInputs:
    def test_with_config_object(self):
        config = MagicMock()
        config.vocab_size = 32000

        inputs = generate_dummy_inputs(config, TaskType.TEXT_CLASSIFICATION)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs


class TestGetDiffusersIOConfig:
    def test_unet(self):
        config = MagicMock()
        config.addition_embed_type = None
        config.time_cond_proj_dim = None
        config.guidance_embeds = False

        io_config = get_diffusers_io_config("unet", config)

        assert "sample" in io_config["input_names"]
        assert "out_sample" in io_config["output_names"]

    def test_unet_sdxl(self):
        config = MagicMock()
        config.addition_embed_type = "text_time"
        config.time_cond_proj_dim = None
        config.guidance_embeds = False

        io_config = get_diffusers_io_config("unet", config)

        assert "text_embeds" in io_config["input_names"]
        assert "time_ids" in io_config["input_names"]

    def test_unet_with_timestep_cond(self):
        config = MagicMock()
        config.addition_embed_type = None
        config.time_cond_proj_dim = 256
        config.guidance_embeds = False

        io_config = get_diffusers_io_config("unet", config)

        assert "timestep_cond" in io_config["input_names"]

    def test_unknown_component_raises(self):
        config = MagicMock()
        with pytest.raises(ValueError, match="Unknown diffusers component"):
            get_diffusers_io_config("unknown", config)
