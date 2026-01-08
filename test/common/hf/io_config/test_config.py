# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
import pytest
from unittest.mock import MagicMock

from olive.common.hf.io_config.config import (
    TextEncoderOnnxConfig,
    TextDecoderOnnxConfig,
    TextDecoderWithPositionIdsOnnxConfig,
    TextSeq2SeqOnnxConfig,
    VisionOnnxConfig,
    TextAndVisionOnnxConfig,
    AudioOnnxConfig,
    AudioToTextOnnxConfig,
)
from olive.common.hf.io_config.tasks import TaskType


class TestTextEncoderOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_dummy_input_generator_classes(self, mock_config):
        """Test DUMMY_INPUT_GENERATOR_CLASSES is set."""
        assert len(TextEncoderOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES) == 1


class TestTextDecoderOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_pad_attention_mask_to_past(self, mock_config):
        """Test PAD_ATTENTION_MASK_TO_PAST is True."""
        assert TextDecoderOnnxConfig.PAD_ATTENTION_MASK_TO_PAST is True

    def test_inputs_without_past(self, mock_config):
        """Test inputs without past key values."""
        onnx_config = TextDecoderOnnxConfig(mock_config, use_past=False)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert not any(k.startswith("past_key_values") for k in inputs)

    def test_inputs_with_past(self, mock_config):
        """Test inputs with past key values."""
        onnx_config = TextDecoderOnnxConfig(mock_config, use_past=True, use_past_in_inputs=True)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert any(k.startswith("past_key_values") for k in inputs)

    def test_outputs_includes_present_when_use_past(self, mock_config):
        """Test outputs include present when use_past is True."""
        onnx_config = TextDecoderOnnxConfig(
            mock_config, task=TaskType.TEXT_GENERATION, use_past=True
        )
        outputs = onnx_config.outputs
        assert any(k.startswith("present") for k in outputs)


class TestTextDecoderWithPositionIdsOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 12
        config.num_attention_heads = 12
        return config

    def test_inputs_includes_position_ids_for_text_generation(self, mock_config):
        """Test inputs include position_ids for text generation task."""
        onnx_config = TextDecoderWithPositionIdsOnnxConfig(
            mock_config, task=TaskType.TEXT_GENERATION
        )
        inputs = onnx_config.inputs
        assert "position_ids" in inputs

    def test_inputs_includes_position_ids_for_feature_extraction(self, mock_config):
        """Test inputs include position_ids for feature extraction task."""
        onnx_config = TextDecoderWithPositionIdsOnnxConfig(
            mock_config, task=TaskType.FEATURE_EXTRACTION
        )
        inputs = onnx_config.inputs
        assert "position_ids" in inputs


class TestTextSeq2SeqOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32128
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.num_attention_heads = 8
        config.d_model = 512
        config.num_heads = 8
        return config

    def test_inputs_include_encoder_and_decoder(self, mock_config):
        """Test inputs include both encoder and decoder inputs."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_ids" in inputs
        assert "attention_mask" in inputs
        assert "decoder_input_ids" in inputs

    def test_inputs_with_past_include_past_key_values(self, mock_config):
        """Test inputs with past include past_key_values."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config, use_past=True, use_past_in_inputs=True)
        inputs = onnx_config.inputs
        assert any(k.startswith("past_key_values") for k in inputs)

    def test_add_past_key_values_includes_encoder_and_decoder(self, mock_config):
        """Test add_past_key_values includes both encoder and decoder entries."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config, use_past=True)
        inputs = {}
        onnx_config.add_past_key_values(inputs, direction="inputs")

        # Should have decoder and encoder entries
        decoder_keys = [k for k in inputs if "decoder" in k]
        encoder_keys = [k for k in inputs if "encoder" in k]
        assert len(decoder_keys) > 0
        assert len(encoder_keys) > 0


class TestVisionOnnxConfig:
    def test_dummy_input_generator_classes(self):
        """Test DUMMY_INPUT_GENERATOR_CLASSES is set."""
        assert len(VisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES) == 1


class TestTextAndVisionOnnxConfig:
    def test_dummy_input_generator_classes(self):
        """Test DUMMY_INPUT_GENERATOR_CLASSES includes text and vision."""
        assert len(TextAndVisionOnnxConfig.DUMMY_INPUT_GENERATOR_CLASSES) == 3


class TestAudioOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.feature_size = 80
        return config

    def test_inputs(self, mock_config):
        """Test inputs returns input_values."""
        onnx_config = AudioOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_values" in inputs
        assert inputs["input_values"] == {0: "batch_size", 1: "sequence_length"}


class TestAudioToTextOnnxConfig:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.num_attention_heads = 8
        config.d_model = 512
        return config

    def test_inputs_include_input_features(self, mock_config):
        """Test inputs include input_features and decoder_input_ids."""
        onnx_config = AudioToTextOnnxConfig(mock_config)
        inputs = onnx_config.inputs
        assert "input_features" in inputs
        assert "decoder_input_ids" in inputs

    def test_add_past_key_values_includes_encoder_decoder(self, mock_config):
        """Test add_past_key_values includes encoder and decoder entries."""
        onnx_config = AudioToTextOnnxConfig(mock_config, use_past=True)
        inputs = {}
        onnx_config.add_past_key_values(inputs, direction="inputs")

        decoder_keys = [k for k in inputs if "decoder" in k]
        encoder_keys = [k for k in inputs if "encoder" in k]
        assert len(decoder_keys) > 0
        assert len(encoder_keys) > 0


class TestTextSeq2SeqOnnxConfigAdvanced:
    """Advanced tests for TextSeq2SeqOnnxConfig."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32128
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.num_attention_heads = 8
        config.d_model = 512
        config.num_heads = 8
        config.encoder_num_layers = 6
        config.decoder_num_layers = 6
        config.encoder_num_attention_heads = 8
        config.decoder_num_attention_heads = 8
        return config

    def test_create_dummy_input_generator_classes(self, mock_config):
        """Test _create_dummy_input_generator_classes creates all generators."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config)
        generators = onnx_config._create_dummy_input_generator_classes()
        assert len(generators) == 3

    def test_generate_dummy_inputs(self, mock_config):
        """Test generate_dummy_inputs creates all required inputs."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config)
        dummy_inputs = onnx_config.generate_dummy_inputs()
        assert "input_ids" in dummy_inputs
        assert "attention_mask" in dummy_inputs
        assert "decoder_input_ids" in dummy_inputs

    def test_add_past_key_values_outputs(self, mock_config):
        """Test add_past_key_values with direction='outputs'."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config, use_past=True)
        outputs = {}
        onnx_config.add_past_key_values(outputs, direction="outputs")

        # Should have present entries
        present_keys = [k for k in outputs if k.startswith("present")]
        assert len(present_keys) > 0
        # Check dynamic axes for decoder
        assert "past_decoder_sequence_length + decoder_sequence_length" in str(outputs)

    def test_add_past_key_values_invalid_direction_raises(self, mock_config):
        """Test add_past_key_values raises for invalid direction."""
        onnx_config = TextSeq2SeqOnnxConfig(mock_config, use_past=True)
        with pytest.raises(ValueError, match="direction must either be"):
            onnx_config.add_past_key_values({}, direction="invalid")


class TestTextDecoderOnnxConfigAdvanced:
    """Advanced tests for TextDecoderOnnxConfig."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 768
        config.num_hidden_layers = 4
        config.num_attention_heads = 12
        return config

    def test_dummy_pkv_generator_class(self, mock_config):
        """Test DUMMY_PKV_GENERATOR_CLASS is set."""
        assert TextDecoderOnnxConfig.DUMMY_PKV_GENERATOR_CLASS is not None

    def test_outputs_is_merged_false(self, mock_config):
        """Test outputs when is_merged is False (default)."""
        onnx_config = TextDecoderOnnxConfig(
            mock_config,
            task=TaskType.TEXT_GENERATION,
            use_past=True,
        )
        onnx_config.is_merged = False
        outputs = onnx_config.outputs
        assert "logits" in outputs
        # When not merged with past in inputs, sequence_length is 1
        # (Only check that the output exists)

    def test_outputs_without_use_past_in_inputs(self, mock_config):
        """Test outputs when use_past_in_inputs is False."""
        onnx_config = TextDecoderOnnxConfig(
            mock_config,
            task=TaskType.TEXT_GENERATION,
            use_past=True,
            use_past_in_inputs=False,
        )
        outputs = onnx_config.outputs
        assert "logits" in outputs
        # Should still have present keys
        present_keys = [k for k in outputs if k.startswith("present")]
        assert len(present_keys) > 0


class TestAudioToTextOnnxConfigAdvanced:
    """Advanced tests for AudioToTextOnnxConfig."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.hidden_size = 512
        config.num_hidden_layers = 6
        config.num_attention_heads = 8
        config.d_model = 512
        config.encoder_num_layers = 6
        config.decoder_num_layers = 6
        config.encoder_num_attention_heads = 8
        config.decoder_num_attention_heads = 8
        return config

    def test_add_past_key_values_outputs(self, mock_config):
        """Test add_past_key_values with direction='outputs'."""
        onnx_config = AudioToTextOnnxConfig(mock_config, use_past=True)
        outputs = {}
        onnx_config.add_past_key_values(outputs, direction="outputs")

        present_keys = [k for k in outputs if k.startswith("present")]
        assert len(present_keys) > 0
