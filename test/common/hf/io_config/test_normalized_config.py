# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest

from olive.common.hf.io_config.normalized_config import (
    BartLikeNormalizedTextConfig,
    GPT2LikeNormalizedTextConfig,
    NormalizedConfig,
    NormalizedConfigManager,
    NormalizedEncoderDecoderConfig,
    NormalizedFluxTransformerConfig,
    NormalizedSanaTransformerConfig,
    NormalizedSD3TransformerConfig,
    NormalizedSegformerConfig,
    NormalizedSeq2SeqConfig,
    NormalizedTextAndVisionConfig,
    NormalizedTextConfig,
    NormalizedTextConfigWithGQA,
    NormalizedUNetConfig,
    NormalizedVaeConfig,
    NormalizedVisionConfig,
    T5LikeNormalizedTextConfig,
)


class TestNormalizedConfig:
    def test_basic_attribute_access(self):
        """Test basic attribute access from config."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12

        normalized = NormalizedConfig(mock_config)
        assert normalized.hidden_size == 768
        assert normalized.num_hidden_layers == 12

    def test_with_args_factory(self):
        """Test with_args class method creates partial."""
        mock_config = MagicMock()
        mock_config.n_embd = 768

        factory = NormalizedConfig.with_args(hidden_size="n_embd", allow_new=True)
        normalized = factory(mock_config)
        assert normalized.hidden_size == 768

    def test_has_attribute_returns_true_for_existing(self):
        """Test has_attribute returns True for existing attributes."""
        mock_config = MagicMock()
        mock_config.vocab_size = 50000

        normalized = NormalizedConfig(mock_config)
        assert normalized.has_attribute("vocab_size") is True

    def test_has_attribute_returns_false_for_missing(self):
        """Test has_attribute returns False for missing attributes."""
        mock_config = MagicMock(spec=[])

        normalized = NormalizedConfig(mock_config)
        assert normalized.has_attribute("nonexistent_attr") is False

    def test_nested_attribute_access(self):
        """Test accessing nested config attributes like 'text_config.hidden_size'."""
        mock_config = MagicMock()
        mock_config.text_config.hidden_size = 512

        normalized = NormalizedConfig(mock_config)
        normalized.HIDDEN_SIZE = "text_config.hidden_size"
        assert normalized.hidden_size == 512


class TestNormalizedTextConfig:
    def test_text_config_attributes(self):
        """Test NormalizedTextConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32
        mock_config.eos_token_id = 2

        normalized = NormalizedTextConfig(mock_config)
        assert normalized.vocab_size == 32000
        assert normalized.hidden_size == 4096
        assert normalized.num_layers == 32
        assert normalized.num_attention_heads == 32
        assert normalized.eos_token_id == 2


class TestNormalizedTextConfigWithGQA:
    def test_gqa_config_has_num_key_value_heads(self):
        """Test GQA config includes num_key_value_heads."""
        mock_config = MagicMock()
        mock_config.num_key_value_heads = 8
        mock_config.vocab_size = 32000
        mock_config.hidden_size = 4096
        mock_config.num_hidden_layers = 32
        mock_config.num_attention_heads = 32

        normalized = NormalizedTextConfigWithGQA(mock_config)
        assert normalized.num_key_value_heads == 8


class TestNormalizedVisionConfig:
    def test_vision_config_attributes(self):
        """Test NormalizedVisionConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.image_size = 224
        mock_config.num_channels = 3

        normalized = NormalizedVisionConfig(mock_config)
        assert normalized.image_size == 224
        assert normalized.num_channels == 3


class TestNormalizedSeq2SeqConfig:
    def test_seq2seq_config_has_encoder_decoder_attrs(self):
        """Test seq2seq config has encoder and decoder attributes."""
        mock_config = MagicMock()
        mock_config.num_hidden_layers = 6
        mock_config.num_attention_heads = 8
        mock_config.hidden_size = 512
        mock_config.vocab_size = 32000

        normalized = NormalizedSeq2SeqConfig(mock_config)
        assert normalized.encoder_num_layers == 6
        assert normalized.decoder_num_layers == 6
        assert normalized.encoder_num_attention_heads == 8
        assert normalized.decoder_num_attention_heads == 8


class TestNormalizedSegformerConfig:
    def test_segformer_returns_zero_for_list_attrs(self):
        """Test segformer returns 0 when attribute is a list."""
        mock_config = MagicMock()
        mock_config.hidden_sizes = [32, 64, 160, 256]
        mock_config.num_attention_heads = [1, 2, 5, 8]

        normalized = NormalizedSegformerConfig(mock_config)
        assert normalized.hidden_size == 0
        assert normalized.num_attention_heads == 0


class TestNormalizedConfigManager:
    def test_get_supported_model_type(self):
        """Test getting config class for supported model type."""
        config_class = NormalizedConfigManager.get_normalized_config_class("bert")
        assert config_class == NormalizedTextConfig

    def test_get_gqa_model_type(self):
        """Test getting config class for GQA model type."""
        config_class = NormalizedConfigManager.get_normalized_config_class("llama")
        assert config_class == NormalizedTextConfigWithGQA

    def test_check_unsupported_model_raises(self):
        """Test that unsupported model type raises KeyError."""
        with pytest.raises(KeyError):
            NormalizedConfigManager.check_supported_model("unsupported_model_xyz")


class TestDiffusersNormalizedConfigs:
    def test_unet_normalized_config(self):
        """Test NormalizedUNetConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.sample_size = 64
        mock_config.in_channels = 4
        mock_config.cross_attention_dim = 768

        factory = NormalizedUNetConfig
        normalized = factory(mock_config)
        assert normalized.vocab_size == 64
        assert normalized.image_size == 64
        assert normalized.num_channels == 4
        assert normalized.hidden_size == 768

    def test_vae_normalized_config(self):
        """Test NormalizedVaeConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.sample_size = 512
        mock_config.in_channels = 3
        mock_config.latent_channels = 4

        factory = NormalizedVaeConfig
        normalized = factory(mock_config)
        assert normalized.image_size == 512
        assert normalized.num_channels == 3
        assert normalized.latent_channels == 4

    def test_sd3_transformer_normalized_config(self):
        """Test NormalizedSD3TransformerConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 16
        mock_config.joint_attention_dim = 4096

        factory = NormalizedSD3TransformerConfig
        normalized = factory(mock_config)
        assert normalized.vocab_size == 128
        assert normalized.image_size == 128
        assert normalized.num_channels == 16
        assert normalized.hidden_size == 4096

    def test_flux_transformer_normalized_config(self):
        """Test NormalizedFluxTransformerConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.sample_size = 128
        mock_config.in_channels = 64
        mock_config.joint_attention_dim = 4096

        factory = NormalizedFluxTransformerConfig
        normalized = factory(mock_config)
        assert normalized.vocab_size == 128
        assert normalized.hidden_size == 4096

    def test_sana_transformer_normalized_config(self):
        """Test NormalizedSanaTransformerConfig attribute mappings."""
        mock_config = MagicMock()
        mock_config.sample_size = 32
        mock_config.in_channels = 32
        mock_config.caption_channels = 2304

        factory = NormalizedSanaTransformerConfig
        normalized = factory(mock_config)
        assert normalized.vocab_size == 32
        assert normalized.hidden_size == 2304


class TestNormalizedTextAndVisionConfig:
    """Test NormalizedTextAndVisionConfig."""

    def test_text_and_vision_config_basic(self):
        """Test basic NormalizedTextAndVisionConfig without TEXT_CONFIG/VISION_CONFIG."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.image_size = 224
        mock_config.num_channels = 3

        normalized = NormalizedTextAndVisionConfig(mock_config)
        assert normalized.hidden_size == 768
        assert normalized.image_size == 224

    def test_text_and_vision_config_with_text_config(self):
        """Test NormalizedTextAndVisionConfig with TEXT_CONFIG set."""
        mock_config = MagicMock()
        mock_config.text_config.hidden_size = 512
        mock_config.text_config.vocab_size = 32000

        # Create subclass with TEXT_CONFIG set
        class TestConfig(NormalizedTextAndVisionConfig):
            TEXT_CONFIG = "text_config"

        normalized = TestConfig(mock_config)
        assert normalized.hidden_size == 512

    def test_text_and_vision_config_with_vision_config(self):
        """Test NormalizedTextAndVisionConfig with VISION_CONFIG set."""
        mock_config = MagicMock()
        mock_config.vision_config.image_size = 384
        mock_config.vision_config.num_channels = 3

        # Create subclass with VISION_CONFIG set
        class TestConfig(NormalizedTextAndVisionConfig):
            VISION_CONFIG = "vision_config"

        normalized = TestConfig(mock_config)
        assert normalized.image_size == 384


class TestNormalizedEncoderDecoderConfig:
    """Test NormalizedEncoderDecoderConfig."""

    def test_encoder_decoder_config_fallback_to_base(self):
        """Test NormalizedEncoderDecoderConfig falls back to base config."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768

        normalized = NormalizedEncoderDecoderConfig(mock_config)
        # With no encoder/decoder class set, should fall back to base __getattr__
        assert normalized.hidden_size == 768

    def test_encoder_decoder_config_with_encoder_class(self):
        """Test NormalizedEncoderDecoderConfig delegates to encoder config class."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.vocab_size = 30000

        # Create encoder config instance that has VOCAB_SIZE attribute
        encoder_config = NormalizedTextConfig(mock_config)

        class TestConfig(NormalizedEncoderDecoderConfig):
            ENCODER_NORMALIZED_CONFIG_CLASS = encoder_config

        normalized = TestConfig(mock_config)
        # vocab_size should delegate to encoder config
        assert normalized.vocab_size == 30000

    def test_encoder_decoder_config_with_decoder_class(self):
        """Test NormalizedEncoderDecoderConfig delegates to decoder config class."""
        mock_config = MagicMock()
        mock_config.hidden_size = 768
        mock_config.num_hidden_layers = 12

        # Create decoder config instance
        decoder_config = NormalizedTextConfig(mock_config)

        class TestConfig(NormalizedEncoderDecoderConfig):
            DECODER_NORMALIZED_CONFIG_CLASS = decoder_config

        normalized = TestConfig(mock_config)
        # num_layers should delegate to decoder config
        assert normalized.num_layers == 12


class TestPreConfiguredNormalizedConfigs:
    def test_gpt2_like_config(self):
        """Test GPT2-like config maps n_embd to hidden_size."""
        mock_config = MagicMock()
        mock_config.n_embd = 768
        mock_config.n_head = 12
        mock_config.vocab_size = 50257
        mock_config.num_hidden_layers = 12

        factory = GPT2LikeNormalizedTextConfig
        normalized = factory(mock_config)
        assert normalized.hidden_size == 768
        assert normalized.num_attention_heads == 12

    def test_bart_like_config(self):
        """Test BART-like config maps d_model to hidden_size."""
        mock_config = MagicMock()
        mock_config.d_model = 1024
        mock_config.encoder_attention_heads = 16
        mock_config.vocab_size = 50265
        mock_config.num_hidden_layers = 12

        factory = BartLikeNormalizedTextConfig
        normalized = factory(mock_config)
        assert normalized.hidden_size == 1024
        assert normalized.num_attention_heads == 16

    def test_t5_like_config(self):
        """Test T5-like config maps d_model to hidden_size."""
        mock_config = MagicMock()
        mock_config.d_model = 512
        mock_config.num_heads = 8
        mock_config.vocab_size = 32128
        mock_config.num_hidden_layers = 6

        factory = T5LikeNormalizedTextConfig
        normalized = factory(mock_config)
        assert normalized.hidden_size == 512
        assert normalized.num_attention_heads == 8
