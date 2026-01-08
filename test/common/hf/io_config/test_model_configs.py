# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.model_configs import (
    _ONNX_CONFIG_REGISTRY,
    BertOnnxConfig,
    # Multimodal
    DcaeDecoderOnnxConfig,
    DcaeEncoderOnnxConfig,
    DiffusersTextEncoderOnnxConfig,
    DiffusersTextEncoderWithProjectionOnnxConfig,
    DistilBertOnnxConfig,
    FluxTransformerOnnxConfig,
    GPT2OnnxConfig,
    LlamaOnnxConfig,
    SanaTransformerOnnxConfig,
    SD3TransformerOnnxConfig,
    # Seq2seq models
    UNetOnnxConfig,
    VaeDecoderOnnxConfig,
    VaeEncoderOnnxConfig,
    # Vision models
    ViTOnnxConfig,
    get_onnx_config_class,
    get_supported_model_types,
    get_supported_tasks_for_model,
)
from olive.common.hf.io_config.tasks import TaskType


class TestRegistry:
    def test_registry_is_populated(self):
        """Test that the registry is populated with model configs."""
        assert len(_ONNX_CONFIG_REGISTRY) > 0
        assert "bert" in _ONNX_CONFIG_REGISTRY
        assert "gpt2" in _ONNX_CONFIG_REGISTRY
        assert "llama" in _ONNX_CONFIG_REGISTRY

    def test_get_onnx_config_class_for_bert(self):
        """Test getting config class for bert."""
        config_class = get_onnx_config_class("bert", TaskType.FEATURE_EXTRACTION)
        assert config_class == BertOnnxConfig

    def test_get_onnx_config_class_unsupported_model_raises(self):
        """Test getting config class for unsupported model raises."""
        with pytest.raises(KeyError, match="not supported"):
            get_onnx_config_class("unsupported_model_xyz", TaskType.FEATURE_EXTRACTION)

    def test_get_onnx_config_class_unsupported_task_raises(self):
        """Test getting config class for unsupported task raises."""
        with pytest.raises(KeyError, match="not supported for model type"):
            get_onnx_config_class("bert", "unsupported-task")

    def test_get_supported_model_types(self):
        """Test get_supported_model_types returns list of models."""
        model_types = get_supported_model_types()
        assert isinstance(model_types, list)
        assert "bert" in model_types
        assert "gpt2" in model_types
        assert "llama" in model_types

    def test_get_supported_tasks_for_model(self):
        """Test get_supported_tasks_for_model returns list of tasks."""
        tasks = get_supported_tasks_for_model("bert")
        assert isinstance(tasks, list)
        assert TaskType.FEATURE_EXTRACTION in tasks
        assert TaskType.TEXT_CLASSIFICATION in tasks

    def test_get_supported_tasks_unsupported_model_raises(self):
        """Test get_supported_tasks_for_model raises for unsupported model."""
        with pytest.raises(KeyError):
            get_supported_tasks_for_model("unsupported_model_xyz")


class TestBertLikeConfigs:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 30522
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.type_vocab_size = 2
        return config

    def test_bert_inputs(self, mock_config):
        """Test BertOnnxConfig inputs include token_type_ids."""
        onnx_config = BertOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "token_type_ids" in inputs

    def test_bert_multiple_choice_shape(self, mock_config):
        """Test BertOnnxConfig inputs have num_choices for MULTIPLE_CHOICE."""
        onnx_config = BertOnnxConfig(mock_config, task=TaskType.MULTIPLE_CHOICE)
        inputs = onnx_config.inputs
        assert inputs["input_ids"][1] == "num_choices"

    def test_distilbert_no_token_type_ids(self, mock_config):
        """Test DistilBertOnnxConfig inputs don't include token_type_ids."""
        onnx_config = DistilBertOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "token_type_ids" not in inputs


class TestGPTLikeConfigs:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 50257
        config.n_embd = 768
        config.n_head = 12
        config.n_layer = 12
        config.num_hidden_layers = 12
        config.hidden_size = 768
        config.num_attention_heads = 12
        return config

    def test_gpt2_has_position_ids(self, mock_config):
        """Test GPT2OnnxConfig includes position_ids for text generation."""
        onnx_config = GPT2OnnxConfig(mock_config, task=TaskType.TEXT_GENERATION)
        inputs = onnx_config.inputs
        assert "position_ids" in inputs


class TestLlamaLikeConfigs:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        return config

    def test_llama_has_position_ids(self, mock_config):
        """Test LlamaOnnxConfig includes position_ids."""
        onnx_config = LlamaOnnxConfig(mock_config, task=TaskType.TEXT_GENERATION)
        inputs = onnx_config.inputs
        assert "position_ids" in inputs


class TestVisionConfigs:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.image_size = 224
        config.num_channels = 3
        return config

    def test_vit_inputs(self, mock_config):
        """Test ViTOnnxConfig inputs."""
        onnx_config = ViTOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "pixel_values" in inputs
        assert inputs["pixel_values"] == {0: "batch_size", 1: "num_channels", 2: "height", 3: "width"}


class TestDiffusersConfigs:
    def test_diffusers_text_encoder_inputs(self):
        """Test DiffusersTextEncoderOnnxConfig inputs."""
        mock_config = MagicMock()
        mock_config.vocab_size = 49408
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12
        mock_config.num_attention_heads = 12

        onnx_config = DiffusersTextEncoderOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert len(inputs) == 1  # Only input_ids

        outputs = onnx_config.outputs
        assert "last_hidden_state" in outputs
        assert "pooler_output" in outputs

    def test_diffusers_text_encoder_with_projection_outputs(self):
        """Test DiffusersTextEncoderWithProjectionOnnxConfig outputs."""
        mock_config = MagicMock()
        mock_config.vocab_size = 49408
        mock_config.hidden_size = 1280
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 20

        onnx_config = DiffusersTextEncoderWithProjectionOnnxConfig(mock_config)
        outputs = onnx_config.outputs
        assert "text_embeds" in outputs
        assert "last_hidden_state" in outputs

    def test_unet_inputs(self):
        """Test UNetOnnxConfig inputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 64
        mock_config.in_channels = 4
        mock_config.cross_attention_dim = 768
        mock_config.addition_embed_type = None
        mock_config.time_cond_proj_dim = None

        onnx_config = UNetOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "sample" in inputs
        assert "timestep" in inputs
        assert "encoder_hidden_states" in inputs
        # Not SDXL, so no text_embeds/time_ids
        assert "text_embeds" not in inputs

    def test_unet_sdxl_inputs(self):
        """Test UNetOnnxConfig inputs for SDXL."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 4
        mock_config.cross_attention_dim = 2048
        mock_config.addition_embed_type = "text_time"
        mock_config.time_cond_proj_dim = None

        onnx_config = UNetOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "text_embeds" in inputs
        assert "time_ids" in inputs

    def test_vae_encoder_inputs_outputs(self):
        """Test VaeEncoderOnnxConfig inputs and outputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 512
        mock_config.in_channels = 3
        mock_config.latent_channels = 4

        onnx_config = VaeEncoderOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "sample" in inputs

        outputs = onnx_config.outputs
        assert "latent_parameters" in outputs

    def test_vae_decoder_inputs_outputs(self):
        """Test VaeDecoderOnnxConfig inputs and outputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 512
        mock_config.in_channels = 3
        mock_config.latent_channels = 4

        onnx_config = VaeDecoderOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "latent_sample" in inputs

        outputs = onnx_config.outputs
        assert "sample" in outputs

    def test_sd3_transformer_inputs(self):
        """Test SD3TransformerOnnxConfig inputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 16
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 2048

        onnx_config = SD3TransformerOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "hidden_states" in inputs
        assert "encoder_hidden_states" in inputs
        assert "pooled_projections" in inputs
        assert "timestep" in inputs

    def test_flux_transformer_inputs(self):
        """Test FluxTransformerOnnxConfig inputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 64
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = True

        onnx_config = FluxTransformerOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "hidden_states" in inputs
        assert "txt_ids" in inputs
        assert "img_ids" in inputs
        assert "guidance" in inputs

    def test_flux_transformer_no_guidance(self):
        """Test FluxTransformerOnnxConfig without guidance."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 64
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = False

        onnx_config = FluxTransformerOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "guidance" not in inputs

    def test_sana_transformer_inputs(self):
        """Test SanaTransformerOnnxConfig inputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 32
        mock_config.in_channels = 32
        mock_config.caption_channels = 2304

        onnx_config = SanaTransformerOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "hidden_states" in inputs
        assert "encoder_hidden_states" in inputs
        assert "encoder_attention_mask" in inputs
        assert "timestep" in inputs

    def test_dcae_encoder_inputs_outputs(self):
        """Test DcaeEncoderOnnxConfig inputs and outputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 512
        mock_config.in_channels = 3
        mock_config.latent_channels = 32

        onnx_config = DcaeEncoderOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "sample" in inputs

        outputs = onnx_config.outputs
        assert "latent" in outputs

    def test_dcae_decoder_inputs_outputs(self):
        """Test DcaeDecoderOnnxConfig inputs and outputs."""
        mock_config = MagicMock()
        mock_config.sample_size = 512
        mock_config.in_channels = 3
        mock_config.latent_channels = 32

        onnx_config = DcaeDecoderOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "latent_sample" in inputs

        outputs = onnx_config.outputs
        assert "sample" in outputs
