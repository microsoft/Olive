# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.model_configs import (
    _ONNX_CONFIG_REGISTRY,
    BertOnnxConfig,
    DistilBertOnnxConfig,
    FluxTransformerOnnxConfig,
    UNetOnnxConfig,
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

    def test_bert_multiple_choice_shape(self, mock_config):
        """Test BertOnnxConfig inputs have num_choices for MULTIPLE_CHOICE."""
        onnx_config = BertOnnxConfig(mock_config, task=TaskType.MULTIPLE_CHOICE)
        inputs = onnx_config.inputs
        assert inputs["input_ids"][1] == "num_choices"

    def test_distilbert_no_token_type_ids(self, mock_config):
        """Test DistilBertOnnxConfig inputs don't include token_type_ids."""
        onnx_config = DistilBertOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "token_type_ids" not in inputs


class TestDiffusersConfigs:
    def test_unet_sdxl_inputs(self):
        """Test UNetOnnxConfig inputs for SDXL include text_embeds and time_ids."""
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
