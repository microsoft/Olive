# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock

from olive.common.hf.io_config import (
    # Main API functions
    get_onnx_config,
    get_onnx_config_class,
    get_supported_model_types,
    get_supported_tasks_for_model,
    is_model_supported,
    is_task_supported,
    # Diffusers API
    get_diffusers_onnx_config,
    get_supported_diffusers_pipelines,
    get_supported_components_for_pipeline,
    # Base classes
    OnnxConfig,
    OnnxConfigWithPast,
    # Task types
    TaskType,
    map_task_synonym,
    # Normalized configs
    NormalizedConfig,
    NormalizedConfigManager,
    # Input generators
    DummyInputGenerator,
    DEFAULT_DUMMY_SHAPES,
)
from olive.common.hf.io_config.model_configs import BertOnnxConfig


class TestGetOnnxConfig:
    @pytest.fixture
    def mock_bert_config(self):
        config = MagicMock()
        config.vocab_size = 30522
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        config.type_vocab_size = 2
        return config

    @pytest.fixture
    def mock_llama_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 4096
        config.num_hidden_layers = 32
        config.num_attention_heads = 32
        config.num_key_value_heads = 8
        return config

    def test_get_onnx_config_returns_correct_class(self, mock_bert_config):
        """Test get_onnx_config returns correct config instance."""
        onnx_config = get_onnx_config(
            model_type="bert",
            task=TaskType.FEATURE_EXTRACTION,
            config=mock_bert_config,
        )
        assert isinstance(onnx_config, BertOnnxConfig)

    def test_get_onnx_config_sets_task(self, mock_bert_config):
        """Test get_onnx_config sets task correctly."""
        onnx_config = get_onnx_config(
            model_type="bert",
            task=TaskType.TEXT_CLASSIFICATION,
            config=mock_bert_config,
        )
        assert onnx_config.task == TaskType.TEXT_CLASSIFICATION

    def test_get_onnx_config_sets_dtypes(self, mock_bert_config):
        """Test get_onnx_config sets dtypes correctly."""
        onnx_config = get_onnx_config(
            model_type="bert",
            task=TaskType.FEATURE_EXTRACTION,
            config=mock_bert_config,
            int_dtype="int32",
            float_dtype="fp16",
        )
        assert onnx_config.int_dtype == "int32"
        assert onnx_config.float_dtype == "fp16"

    def test_get_onnx_config_with_past_for_decoder(self, mock_llama_config):
        """Test get_onnx_config sets use_past for decoder models."""
        onnx_config = get_onnx_config(
            model_type="llama",
            task=TaskType.TEXT_GENERATION,
            config=mock_llama_config,
            use_past=True,
            use_past_in_inputs=True,
        )
        assert onnx_config.use_past is True
        assert onnx_config.use_past_in_inputs is True


class TestIsModelSupported:
    def test_supported_model_returns_true(self):
        """Test is_model_supported returns True for supported model."""
        assert is_model_supported("bert") is True
        assert is_model_supported("llama") is True
        assert is_model_supported("gpt2") is True

    def test_unsupported_model_returns_false(self):
        """Test is_model_supported returns False for unsupported model."""
        assert is_model_supported("unsupported_model_xyz") is False


class TestIsTaskSupported:
    def test_supported_task_returns_true(self):
        """Test is_task_supported returns True for supported task."""
        assert is_task_supported("bert", TaskType.FEATURE_EXTRACTION) is True
        assert is_task_supported("bert", TaskType.TEXT_CLASSIFICATION) is True

    def test_unsupported_task_returns_false(self):
        """Test is_task_supported returns False for unsupported task."""
        assert is_task_supported("bert", "unsupported-task") is False

    def test_unsupported_model_returns_false(self):
        """Test is_task_supported returns False for unsupported model."""
        assert is_task_supported("unsupported_model", TaskType.FEATURE_EXTRACTION) is False


class TestDiffusersAPI:
    @pytest.fixture
    def mock_clip_config(self):
        config = MagicMock()
        config.vocab_size = 49408
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    @pytest.fixture
    def mock_unet_config(self):
        config = MagicMock()
        config.sample_size = 64
        config.in_channels = 4
        config.cross_attention_dim = 768
        config.addition_embed_type = None
        config.time_cond_proj_dim = None
        return config

    @pytest.fixture
    def mock_vae_config(self):
        config = MagicMock()
        config.sample_size = 512
        config.in_channels = 3
        config.latent_channels = 4
        return config

    def test_get_diffusers_onnx_config_sd_text_encoder(self, mock_clip_config):
        """Test get_diffusers_onnx_config for SD text_encoder."""
        onnx_config = get_diffusers_onnx_config(
            pipeline_type="sd",
            component_name="text_encoder",
            config=mock_clip_config,
        )
        assert "input_ids" in onnx_config.inputs

    def test_get_diffusers_onnx_config_sd_unet(self, mock_unet_config):
        """Test get_diffusers_onnx_config for SD unet."""
        onnx_config = get_diffusers_onnx_config(
            pipeline_type="sd",
            component_name="unet",
            config=mock_unet_config,
        )
        assert "sample" in onnx_config.inputs
        assert "timestep" in onnx_config.inputs

    def test_get_diffusers_onnx_config_sd_vae_encoder(self, mock_vae_config):
        """Test get_diffusers_onnx_config for SD vae_encoder."""
        onnx_config = get_diffusers_onnx_config(
            pipeline_type="sd",
            component_name="vae_encoder",
            config=mock_vae_config,
        )
        assert "sample" in onnx_config.inputs

    def test_get_diffusers_onnx_config_unsupported_raises(self, mock_clip_config):
        """Test get_diffusers_onnx_config raises for unsupported combination."""
        with pytest.raises(KeyError, match="not supported"):
            get_diffusers_onnx_config(
                pipeline_type="unsupported_pipeline",
                component_name="text_encoder",
                config=mock_clip_config,
            )

    def test_get_supported_diffusers_pipelines(self):
        """Test get_supported_diffusers_pipelines returns list."""
        pipelines = get_supported_diffusers_pipelines()
        assert isinstance(pipelines, list)
        assert "sd" in pipelines
        assert "sdxl" in pipelines
        assert "sd3" in pipelines
        assert "flux" in pipelines
        assert "sana" in pipelines

    def test_get_supported_components_for_pipeline(self):
        """Test get_supported_components_for_pipeline returns components."""
        sd_components = get_supported_components_for_pipeline("sd")
        assert "text_encoder" in sd_components
        assert "unet" in sd_components
        assert "vae_encoder" in sd_components
        assert "vae_decoder" in sd_components

        sdxl_components = get_supported_components_for_pipeline("sdxl")
        assert "text_encoder_2" in sdxl_components

        sd3_components = get_supported_components_for_pipeline("sd3")
        assert "transformer" in sd3_components
        assert "text_encoder_3" in sd3_components


class TestExports:
    def test_all_exports_are_importable(self):
        """Test that all __all__ exports are importable."""
        from olive.common.hf.io_config import __all__

        for name in __all__:
            # Just check it exists in the module
            assert hasattr(__import__("olive.common.hf.io_config", fromlist=[name]), name)

    def test_base_classes_exported(self):
        """Test base classes are exported."""
        assert OnnxConfig is not None
        assert OnnxConfigWithPast is not None

    def test_task_types_exported(self):
        """Test task types are exported."""
        assert TaskType is not None
        assert map_task_synonym is not None

    def test_normalized_config_exported(self):
        """Test normalized config classes are exported."""
        assert NormalizedConfig is not None
        assert NormalizedConfigManager is not None

    def test_input_generator_exported(self):
        """Test input generator is exported."""
        assert DummyInputGenerator is not None
        assert DEFAULT_DUMMY_SHAPES is not None


class TestDiffusersConfigRegistry:
    def test_sd_pipeline_components(self):
        """Test SD pipeline has all expected components."""
        components = get_supported_components_for_pipeline("sd")
        expected = ["text_encoder", "unet", "vae_encoder", "vae_decoder"]
        for comp in expected:
            assert comp in components

    def test_sdxl_pipeline_components(self):
        """Test SDXL pipeline has all expected components."""
        components = get_supported_components_for_pipeline("sdxl")
        expected = ["text_encoder", "text_encoder_2", "unet", "vae_encoder", "vae_decoder"]
        for comp in expected:
            assert comp in components

    def test_sd3_pipeline_components(self):
        """Test SD3 pipeline has all expected components."""
        components = get_supported_components_for_pipeline("sd3")
        expected = ["text_encoder", "text_encoder_2", "text_encoder_3", "transformer", "vae_encoder", "vae_decoder"]
        for comp in expected:
            assert comp in components

    def test_flux_pipeline_components(self):
        """Test Flux pipeline has all expected components."""
        components = get_supported_components_for_pipeline("flux")
        expected = ["text_encoder", "text_encoder_2", "transformer", "vae_encoder", "vae_decoder"]
        for comp in expected:
            assert comp in components

    def test_sana_pipeline_components(self):
        """Test Sana pipeline has all expected components."""
        components = get_supported_components_for_pipeline("sana")
        expected = ["text_encoder", "transformer", "vae_encoder", "vae_decoder"]
        for comp in expected:
            assert comp in components
