# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.input_generators import (
    DiffusersDummyInputGenerator,
    TaskDummyInputGenerator,
    _generate_past_key_values,
    generate_diffusers_dummy_inputs,
    generate_task_dummy_inputs,
)
from olive.common.hf.io_config.tasks import TaskType
from olive.constants import DiffusersComponent


class TestGenerateTaskDummyInputs:
    def test_text_classification(self):
        config = MagicMock()
        config.vocab_size = 32000

        inputs = generate_task_dummy_inputs(TaskType.TEXT_CLASSIFICATION, config)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert inputs["input_ids"].shape == (2, 16)

    def test_text_generation(self):
        config = MagicMock()
        config.vocab_size = 32000

        inputs = generate_task_dummy_inputs(TaskType.TEXT_GENERATION, config)

        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "position_ids" in inputs

    def test_multiple_choice(self):
        config = MagicMock()
        config.vocab_size = 32000

        inputs = generate_task_dummy_inputs(TaskType.MULTIPLE_CHOICE, config)

        assert inputs["input_ids"].shape == (2, 4, 16)  # batch, num_choices, seq

    def test_with_past_key_values(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.hidden_size = 64
        config.num_key_value_heads = 4
        config.multi_query = False
        config.query_pre_attn_scalar = None

        inputs = generate_task_dummy_inputs(TaskType.TEXT_GENERATION, config, use_past=True, use_past_in_inputs=True)

        assert "past_key_values" in inputs
        assert len(inputs["past_key_values"]) == 2  # num_layers

    def test_with_model_signature(self):
        config = MagicMock()
        config.vocab_size = 32000

        # Mock model with forward signature
        model = MagicMock()
        model.forward = lambda self, input_ids, attention_mask: None

        inputs = generate_task_dummy_inputs(TaskType.TEXT_CLASSIFICATION, config, model=model)

        assert "input_ids" in inputs

    def test_unknown_task_raises(self):
        config = MagicMock()
        with pytest.raises(ValueError, match="Unknown task"):
            generate_task_dummy_inputs("unknown-task", config)


class TestGenerateDiffusersDummyInputs:
    def test_unet(self):
        config = MagicMock()
        config.in_channels = 4
        config.sample_size = 64
        config.cross_attention_dim = 768

        inputs = generate_diffusers_dummy_inputs(DiffusersComponent.UNET, config)

        assert "sample" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs
        assert inputs["sample"].shape == (2, 4, 64, 64)

    def test_flux_with_guidance(self):
        config = MagicMock()
        config.in_channels = 64
        config.joint_attention_dim = 4096
        config.pooled_projection_dim = 768
        config.guidance_embeds = True

        inputs = generate_diffusers_dummy_inputs(DiffusersComponent.FLUX_TRANSFORMER, config)

        assert "hidden_states" in inputs
        assert "guidance" in inputs

    def test_sd3_transformer(self):
        config = MagicMock()
        config.in_channels = 16
        config.joint_attention_dim = 4096
        config.pooled_projection_dim = 2048

        inputs = generate_diffusers_dummy_inputs(DiffusersComponent.SD3_TRANSFORMER, config)

        assert "hidden_states" in inputs
        assert "pooled_projections" in inputs

    def test_unknown_component_raises(self):
        config = MagicMock()
        with pytest.raises(ValueError, match="Unknown diffusers component"):
            generate_diffusers_dummy_inputs("unknown", config)


class TestTaskDummyInputGenerator:
    def test_supports_input(self):
        config = MagicMock()
        gen = TaskDummyInputGenerator(TaskType.TEXT_CLASSIFICATION, config)

        assert gen.supports_input("input_ids") is True
        assert gen.supports_input("unknown_input") is False

    def test_supported_input_names(self):
        config = MagicMock()
        gen = TaskDummyInputGenerator(TaskType.TEXT_CLASSIFICATION, config)

        names = gen.supported_input_names
        assert "input_ids" in names
        assert "attention_mask" in names

    def test_nested_config_attribute(self):
        """Test that nested config attributes like vision_config.image_size work."""
        config = MagicMock()
        config.vision_config.image_size = 224
        config.vision_config.num_channels = 3

        inputs = generate_task_dummy_inputs(TaskType.ZERO_SHOT_IMAGE_CLASSIFICATION, config)

        assert "pixel_values" in inputs
        # Shape should be [batch, channels, height, width] = [2, 3, 224, 224]
        assert inputs["pixel_values"].shape == (2, 3, 224, 224)


class TestDiffusersDummyInputGenerator:
    def test_supports_input(self):
        config = MagicMock()
        config.in_channels = 4
        config.sample_size = 64
        config.cross_attention_dim = 768

        gen = DiffusersDummyInputGenerator(DiffusersComponent.UNET, config)

        assert gen.supports_input("sample") is True
        assert gen.supports_input("unknown") is False

    def test_supported_input_names(self):
        config = MagicMock()
        config.in_channels = 4
        config.sample_size = 64
        config.cross_attention_dim = 768

        gen = DiffusersDummyInputGenerator(DiffusersComponent.UNET, config)

        names = gen.supported_input_names
        assert "sample" in names

    def test_config_mapping_with_list(self):
        config = MagicMock()
        config.in_channels = [4, 8]  # list value
        config.sample_size = 64
        config.cross_attention_dim = 768

        gen = DiffusersDummyInputGenerator(DiffusersComponent.UNET, config)
        sample = gen.generate("sample")

        assert sample.shape[1] == 4  # should use first element


class TestGeneratePastKeyValues:
    def test_basic(self):
        class Config:
            num_hidden_layers = 2
            num_attention_heads = 4
            hidden_size = 64

        pkv = _generate_past_key_values(Config())

        assert len(pkv) == 2  # num_layers
        assert pkv[0][0].shape == (2, 4, 16, 16)  # batch, heads, seq, head_dim

    def test_with_gqa(self):
        class Config:
            num_hidden_layers = 2
            num_attention_heads = 8
            hidden_size = 64
            num_key_value_heads = 2  # GQA

        pkv = _generate_past_key_values(Config())

        assert pkv[0][0].shape[1] == 2  # num_kv_heads

    def test_with_mqa(self):
        class Config:
            num_hidden_layers = 2
            num_attention_heads = 8
            hidden_size = 64
            multi_query = True

        pkv = _generate_past_key_values(Config())

        assert pkv[0][0].shape[1] == 1  # MQA uses 1 head
