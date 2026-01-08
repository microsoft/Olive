# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest
import torch

from olive.common.hf.io_config.input_generators import (
    DEFAULT_DUMMY_SHAPES,
    DtypeMapper,
    DummyAudioInputGenerator,
    DummyBboxInputGenerator,
    DummyFluxTransformerInputGenerator,
    DummyInputGenerator,
    DummyPastKeyValuesGenerator,
    DummySanaTransformerInputGenerator,
    DummySD3TransformerInputGenerator,
    DummySeq2SeqDecoderTextInputGenerator,
    DummySeq2SeqPastKeyValuesGenerator,
    DummyTextInputGenerator,
    # Diffusers generators
    DummyTimestepInputGenerator,
    DummyUNetInputGenerator,
    DummyVaeInputGenerator,
    DummyVisionInputGenerator,
    FalconDummyPastKeyValuesGenerator,
    GemmaDummyPastKeyValuesGenerator,
    GPTBigCodeDummyPastKeyValuesGenerator,
    MistralDummyPastKeyValuesGenerator,
)
from olive.common.hf.io_config.tasks import TaskType


class TestDTYPEMapper:
    def test_pt_dtype_mapping(self):
        """Test PyTorch dtype mapping."""
        assert DtypeMapper.pt("fp32") == torch.float32
        assert DtypeMapper.pt("fp16") == torch.float16
        assert DtypeMapper.pt("bf16") == torch.bfloat16
        assert DtypeMapper.pt("int64") == torch.int64
        assert DtypeMapper.pt("int32") == torch.int32

    def test_np_dtype_mapping(self):
        """Test NumPy dtype mapping."""
        import numpy as np

        assert DtypeMapper.np("fp32") == np.float32
        assert DtypeMapper.np("fp16") == np.float16
        assert DtypeMapper.np("int64") == np.int64
        assert DtypeMapper.np("int32") == np.int32


class TestDummyInputGeneratorStatic:
    def test_random_int_tensor_shape(self):
        """Test random_int_tensor generates correct shape."""
        tensor = DummyInputGenerator.random_int_tensor([2, 16], max_value=100)
        assert tensor.shape == (2, 16)
        assert tensor.dtype == torch.int64

    def test_random_int_tensor_range(self):
        """Test random_int_tensor values are in range."""
        tensor = DummyInputGenerator.random_int_tensor([10, 10], max_value=50, min_value=10)
        assert (tensor >= 10).all()
        assert (tensor < 50).all()

    def test_random_float_tensor_shape(self):
        """Test random_float_tensor generates correct shape."""
        tensor = DummyInputGenerator.random_float_tensor([2, 3, 64, 64])
        assert tensor.shape == (2, 3, 64, 64)
        assert tensor.dtype == torch.float32

    def test_random_mask_tensor_shape(self):
        """Test random_mask_tensor generates correct shape."""
        tensor = DummyInputGenerator.random_mask_tensor([2, 16])
        assert tensor.shape == (2, 16)
        assert tensor.dtype == torch.int64

    def test_constant_tensor_value(self):
        """Test constant_tensor generates correct values."""
        tensor = DummyInputGenerator.constant_tensor([2, 2], value=5)
        assert (tensor == 5).all()

    def test_pad_input_on_dim(self):
        """Test pad_input_on_dim pads correctly."""
        input_tensor = torch.ones(2, 10)
        padded = DummyInputGenerator.pad_input_on_dim(input_tensor, dim=1, desired_length=15, value=0)
        assert padded.shape == (2, 15)
        assert (padded[:, :10] == 1).all()
        assert (padded[:, 10:] == 0).all()

    def test_concat_inputs(self):
        """Test concat_inputs concatenates correctly."""
        t1 = torch.ones(2, 5)
        t2 = torch.zeros(2, 5)
        result = DummyInputGenerator.concat_inputs([t1, t2], dim=1)
        assert result.shape == (2, 10)


class TestDummyTextInputGenerator:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        return config

    def test_generates_input_ids(self, mock_config):
        """Test generating input_ids."""
        gen = DummyTextInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        input_ids = gen.generate("input_ids")
        assert input_ids.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], DEFAULT_DUMMY_SHAPES["sequence_length"])
        assert input_ids.dtype == torch.int64

    def test_generates_attention_mask(self, mock_config):
        """Test generating attention_mask."""
        gen = DummyTextInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        mask = gen.generate("attention_mask")
        assert mask.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], DEFAULT_DUMMY_SHAPES["sequence_length"])

    def test_generates_position_ids(self, mock_config):
        """Test generating position_ids."""
        gen = DummyTextInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        pos_ids = gen.generate("position_ids")
        assert pos_ids.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], DEFAULT_DUMMY_SHAPES["sequence_length"])

    def test_multiple_choice_shape(self, mock_config):
        """Test shape for multiple choice task."""
        gen = DummyTextInputGenerator(TaskType.MULTIPLE_CHOICE, mock_config)
        input_ids = gen.generate("input_ids")
        expected_shape = (
            DEFAULT_DUMMY_SHAPES["batch_size"],
            DEFAULT_DUMMY_SHAPES["num_choices"],
            DEFAULT_DUMMY_SHAPES["sequence_length"],
        )
        assert input_ids.shape == expected_shape

    def test_supports_input(self, mock_config):
        """Test supports_input returns True for supported inputs."""
        gen = DummyTextInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        assert gen.supports_input("input_ids") is True
        assert gen.supports_input("attention_mask") is True
        assert gen.supports_input("pixel_values") is False


class TestDummyVisionInputGenerator:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.image_size = 224
        config.num_channels = 3
        return config

    def test_generates_pixel_values(self, mock_config):
        """Test generating pixel_values."""
        mock_config.has_attribute = lambda x: x in ["image_size", "num_channels"]
        gen = DummyVisionInputGenerator(TaskType.IMAGE_CLASSIFICATION, mock_config)
        pixels = gen.generate("pixel_values")
        assert pixels.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 3, 224, 224)
        assert pixels.dtype == torch.float32

    def test_generates_pixel_mask(self, mock_config):
        """Test generating pixel_mask."""
        mock_config.has_attribute = lambda x: x in ["image_size", "num_channels"]
        gen = DummyVisionInputGenerator(TaskType.IMAGE_CLASSIFICATION, mock_config)
        mask = gen.generate("pixel_mask")
        assert mask.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 224, 224)


class TestDummyPastKeyValuesGenerator:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.num_layers = 12
        config.num_attention_heads = 12
        config.hidden_size = 768
        return config

    def test_generates_past_key_values(self, mock_config):
        """Test generating past_key_values."""
        gen = DummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 12
        assert len(pkv[0]) == 2
        key, value = pkv[0]
        assert key.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]
        assert key.shape[1] == 12  # num_attention_heads


class TestDummyAudioInputGenerator:
    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.feature_size = 80
        return config

    def test_generates_input_features(self, mock_config):
        """Test generating input_features."""
        gen = DummyAudioInputGenerator(TaskType.AUTOMATIC_SPEECH_RECOGNITION, mock_config)
        features = gen.generate("input_features")
        assert features.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]
        assert features.shape[1] == 80

    def test_generates_input_values(self, mock_config):
        """Test generating input_values (raw waveform)."""
        gen = DummyAudioInputGenerator(TaskType.AUTOMATIC_SPEECH_RECOGNITION, mock_config)
        values = gen.generate("input_values")
        assert values.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]


class TestDiffusersInputGenerators:
    def test_timestep_generator(self):
        """Test DummyTimestepInputGenerator."""
        mock_config = MagicMock()
        mock_config.vocab_size = 1000
        mock_config.text_encoder_projection_dim = 1280
        mock_config.time_cond_proj_dim = 256

        gen = DummyTimestepInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        timestep = gen.generate("timestep")
        assert timestep.shape == ()  # scalar
        assert timestep.dtype == torch.float32

        text_embeds = gen.generate("text_embeds")
        assert text_embeds.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 1280)

    def test_unet_input_generator(self):
        """Test DummyUNetInputGenerator."""
        mock_config = MagicMock()
        mock_config.in_channels = 4
        mock_config.sample_size = 64
        mock_config.cross_attention_dim = 768

        gen = DummyUNetInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        sample = gen.generate("sample")
        assert sample.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 4, 64, 64)

        hidden_states = gen.generate("encoder_hidden_states")
        assert hidden_states.shape == (
            DEFAULT_DUMMY_SHAPES["batch_size"],
            DEFAULT_DUMMY_SHAPES["sequence_length"],
            768,
        )

    def test_vae_input_generator(self):
        """Test DummyVaeInputGenerator."""
        mock_config = MagicMock()
        mock_config.in_channels = 3
        mock_config.latent_channels = 4
        mock_config.sample_size = 512
        mock_config.down_block_types = ["DownEncoderBlock2D"] * 4

        gen = DummyVaeInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        sample = gen.generate("sample")
        assert sample.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 3, 512, 512)

        latent = gen.generate("latent_sample")
        assert latent.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]
        assert latent.shape[1] == 4  # latent_channels

    def test_sd3_transformer_generator(self):
        """Test DummySD3TransformerInputGenerator."""
        mock_config = MagicMock()
        mock_config.in_channels = 16
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 2048

        gen = DummySD3TransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        hidden = gen.generate("hidden_states")
        assert hidden.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 16, 128, 128)

        pooled = gen.generate("pooled_projections")
        assert pooled.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 2048)

        timestep = gen.generate("timestep")
        assert timestep.shape == (DEFAULT_DUMMY_SHAPES["batch_size"],)

    def test_flux_transformer_generator(self):
        """Test DummyFluxTransformerInputGenerator."""
        mock_config = MagicMock()
        mock_config.in_channels = 64
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = True

        gen = DummyFluxTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)

        hidden = gen.generate("hidden_states")
        expected_packed = (128 * 128) // 4
        assert hidden.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], expected_packed, 64)

        txt_ids = gen.generate("txt_ids")
        assert txt_ids.shape[1] == 3

        img_ids = gen.generate("img_ids")
        assert img_ids.shape == (expected_packed, 3)

        guidance = gen.generate("guidance")
        assert guidance.shape == (DEFAULT_DUMMY_SHAPES["batch_size"],)

    def test_sana_transformer_generator(self):
        """Test DummySanaTransformerInputGenerator."""
        mock_config = MagicMock()
        mock_config.in_channels = 32
        mock_config.sample_size = 32
        mock_config.caption_channels = 2304

        gen = DummySanaTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        hidden = gen.generate("hidden_states")
        assert hidden.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 32, 32, 32)

        encoder_mask = gen.generate("encoder_attention_mask")
        assert encoder_mask.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], DEFAULT_DUMMY_SHAPES["sequence_length"])


class TestSpecializedPKVGenerators:
    def test_mistral_pkv_generator(self):
        """Test MistralDummyPastKeyValuesGenerator uses num_key_value_heads."""
        mock_config = MagicMock()
        mock_config.num_layers = 32
        mock_config.num_attention_heads = 32
        mock_config.hidden_size = 4096
        mock_config.num_key_value_heads = 8
        mock_config.head_dim = 128

        gen = MistralDummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 32
        key, value = pkv[0]
        assert key.shape[1] == 8  # num_key_value_heads, not num_attention_heads

    def test_falcon_pkv_generator(self):
        """Test FalconDummyPastKeyValuesGenerator."""
        mock_config = MagicMock()
        mock_config.num_layers = 32
        mock_config.num_attention_heads = 32
        mock_config.hidden_size = 4096
        mock_config.num_kv_heads = 8
        mock_config.new_decoder_architecture = True
        mock_config.multi_query = False

        gen = FalconDummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 32

    def test_gpt_bigcode_pkv_generator_multi_query(self):
        """Test GPTBigCodeDummyPastKeyValuesGenerator with multi_query."""
        mock_config = MagicMock()
        mock_config.num_layers = 40
        mock_config.num_attention_heads = 16
        mock_config.hidden_size = 6144
        mock_config.multi_query = True

        gen = GPTBigCodeDummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 40
        key, value = pkv[0]
        assert key.shape[1] == 1  # multi_query uses 1 head

    def test_gemma_pkv_generator(self):
        """Test GemmaDummyPastKeyValuesGenerator."""
        mock_config = MagicMock()
        mock_config.num_layers = 18
        mock_config.num_attention_heads = 8
        mock_config.hidden_size = 2048
        mock_config.num_key_value_heads = 1
        mock_config.head_dim = 256

        gen = GemmaDummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 18
        key, value = pkv[0]
        assert key.shape[1] == 1  # num_key_value_heads


class TestRandomRanges:
    """Test random batch size and sequence length ranges."""

    def test_text_generator_random_batch_size(self):
        """Test DummyTextInputGenerator with random batch size range."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000

        gen = DummyTextInputGenerator(
            TaskType.FEATURE_EXTRACTION,
            mock_config,
            random_batch_size_range=(1, 4),
        )
        input_ids = gen.generate("input_ids")
        assert 1 <= input_ids.shape[0] <= 4

    def test_text_generator_random_sequence_length(self):
        """Test DummyTextInputGenerator with random sequence length range."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000

        gen = DummyTextInputGenerator(
            TaskType.FEATURE_EXTRACTION,
            mock_config,
            random_sequence_length_range=(8, 32),
        )
        input_ids = gen.generate("input_ids")
        assert 8 <= input_ids.shape[1] <= 32

    def test_text_generator_random_num_choices(self):
        """Test DummyTextInputGenerator with random num_choices range."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000

        gen = DummyTextInputGenerator(
            TaskType.MULTIPLE_CHOICE,
            mock_config,
            random_num_choices_range=(2, 5),
        )
        input_ids = gen.generate("input_ids")
        assert 2 <= input_ids.shape[1] <= 5

    def test_pkv_generator_random_ranges(self):
        """Test DummyPastKeyValuesGenerator with random ranges."""
        mock_config = MagicMock()
        mock_config.num_layers = 4
        mock_config.num_attention_heads = 8
        mock_config.hidden_size = 512

        gen = DummyPastKeyValuesGenerator(
            TaskType.TEXT_GENERATION,
            mock_config,
            random_batch_size_range=(1, 4),
            random_sequence_length_range=(8, 16),
        )
        pkv = gen.generate("past_key_values")
        key, _ = pkv[0]
        assert 1 <= key.shape[0] <= 4
        assert 8 <= key.shape[2] <= 16


class TestSeq2SeqGenerators:
    """Test Seq2Seq-specific generators."""

    @pytest.fixture
    def mock_seq2seq_config(self):
        config = MagicMock()
        config.vocab_size = 32000
        config.hidden_size = 512
        config.encoder_num_layers = 6
        config.decoder_num_layers = 6
        config.encoder_num_attention_heads = 8
        config.decoder_num_attention_heads = 8
        return config

    def test_seq2seq_decoder_text_generator(self, mock_seq2seq_config):
        """Test DummySeq2SeqDecoderTextInputGenerator."""
        gen = DummySeq2SeqDecoderTextInputGenerator(
            TaskType.TEXT2TEXT_GENERATION,
            mock_seq2seq_config,
        )
        decoder_input_ids = gen.generate("decoder_input_ids")
        assert decoder_input_ids.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]

        encoder_outputs = gen.generate("encoder_hidden_states")
        assert isinstance(encoder_outputs, tuple)
        assert len(encoder_outputs) == 3

    def test_seq2seq_pkv_generator(self, mock_seq2seq_config):
        """Test DummySeq2SeqPastKeyValuesGenerator."""
        gen = DummySeq2SeqPastKeyValuesGenerator(
            TaskType.TEXT2TEXT_GENERATION,
            mock_seq2seq_config,
        )
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 6
        # Seq2seq PKV has 4 elements per layer (decoder key/value, encoder key/value)
        assert len(pkv[0]) == 4

    def test_seq2seq_pkv_generator_cache_position(self, mock_seq2seq_config):
        """Test DummySeq2SeqPastKeyValuesGenerator cache_position input."""
        gen = DummySeq2SeqPastKeyValuesGenerator(
            TaskType.TEXT2TEXT_GENERATION,
            mock_seq2seq_config,
        )
        cache_pos = gen.generate("cache_position")
        assert cache_pos.shape == (1,)

    def test_seq2seq_pkv_generator_unsupported_raises(self, mock_seq2seq_config):
        """Test DummySeq2SeqPastKeyValuesGenerator raises for unsupported input."""
        gen = DummySeq2SeqPastKeyValuesGenerator(
            TaskType.TEXT2TEXT_GENERATION,
            mock_seq2seq_config,
        )
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")

    def test_seq2seq_pkv_generator_with_random_ranges(self, mock_seq2seq_config):
        """Test DummySeq2SeqPastKeyValuesGenerator with random ranges."""
        gen = DummySeq2SeqPastKeyValuesGenerator(
            TaskType.TEXT2TEXT_GENERATION,
            mock_seq2seq_config,
            random_batch_size_range=(1, 4),
            random_sequence_length_range=(8, 16),
        )
        pkv = gen.generate("past_key_values")
        decoder_key, _, _, _ = pkv[0]
        assert 1 <= decoder_key.shape[0] <= 4


class TestBboxGenerator:
    """Test DummyBboxInputGenerator."""

    def test_bbox_generator(self):
        """Test DummyBboxInputGenerator generates bbox."""
        mock_config = MagicMock()
        mock_config.vocab_size = 32000

        gen = DummyBboxInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        bbox = gen.generate("bbox")
        assert bbox.shape == (
            DEFAULT_DUMMY_SHAPES["batch_size"],
            DEFAULT_DUMMY_SHAPES["sequence_length"],
            4,
        )
        assert bbox.dtype == torch.int64


class TestVisionGeneratorAdditional:
    """Additional tests for DummyVisionInputGenerator."""

    @pytest.fixture
    def mock_config(self):
        config = MagicMock()
        config.image_size = 224
        config.num_channels = 3
        config.has_attribute = lambda x: x in ["image_size", "num_channels"]
        return config

    def test_vision_generator_visual_embeds(self, mock_config):
        """Test generating visual_embeds."""
        mock_config.visual_embedding_dim = 512
        mock_config.has_attribute = lambda x: x in ["image_size", "num_channels", "visual_embedding_dim"]
        gen = DummyVisionInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        embeds = gen.generate("visual_embeds")
        assert embeds.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]

    def test_vision_generator_visual_attention_mask(self, mock_config):
        """Test generating visual_attention_mask."""
        gen = DummyVisionInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        mask = gen.generate("visual_attention_mask")
        assert mask.dtype == torch.int64

    def test_vision_generator_visual_token_type_ids(self, mock_config):
        """Test generating visual_token_type_ids."""
        gen = DummyVisionInputGenerator(TaskType.FEATURE_EXTRACTION, mock_config)
        token_ids = gen.generate("visual_token_type_ids")
        assert token_ids.dtype == torch.int64


class TestNumpyFramework:
    """Test numpy framework support."""

    def test_random_int_tensor_numpy(self):
        """Test random_int_tensor with numpy framework."""
        import numpy as np

        tensor = DummyInputGenerator.random_int_tensor([2, 16], max_value=100, framework="np")
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (2, 16)
        assert tensor.dtype == np.int64

    def test_random_float_tensor_numpy(self):
        """Test random_float_tensor with numpy framework."""
        import numpy as np

        tensor = DummyInputGenerator.random_float_tensor([2, 3, 64, 64], framework="np")
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (2, 3, 64, 64)
        assert tensor.dtype == np.float32

    def test_random_mask_tensor_numpy(self):
        """Test random_mask_tensor with numpy framework."""
        import numpy as np

        tensor = DummyInputGenerator.random_mask_tensor([2, 16], framework="np")
        assert isinstance(tensor, np.ndarray)
        assert tensor.shape == (2, 16)

    def test_constant_tensor_numpy(self):
        """Test constant_tensor with numpy framework."""
        import numpy as np

        tensor = DummyInputGenerator.constant_tensor([2, 2], value=5, framework="np")
        assert isinstance(tensor, np.ndarray)
        assert (tensor == 5).all()


class TestConcatInputsEdgeCases:
    """Test edge cases for concat_inputs."""

    def test_concat_inputs_empty_raises(self):
        """Test concat_inputs with empty list raises ValueError."""
        with pytest.raises(ValueError, match="You did not provide any inputs"):
            DummyInputGenerator.concat_inputs([], dim=0)

    def test_concat_inputs_numpy(self):
        """Test concat_inputs with numpy arrays."""
        import numpy as np

        t1 = np.ones((2, 5))
        t2 = np.zeros((2, 5))
        result = DummyInputGenerator.concat_inputs([t1, t2], dim=1)
        assert isinstance(result, np.ndarray)
        assert result.shape == (2, 10)


class TestPadInputEdgeCases:
    """Test edge cases for pad_input_on_dim."""

    def test_pad_requires_one_length_param(self):
        """Test pad_input_on_dim requires exactly one of desired_length or padding_length."""
        tensor = torch.ones(2, 10)
        with pytest.raises(ValueError, match="You need to provide either"):
            DummyInputGenerator.pad_input_on_dim(tensor, dim=1)

    def test_pad_requires_only_one_length_param(self):
        """Test pad_input_on_dim rejects both desired_length and padding_length."""
        tensor = torch.ones(2, 10)
        with pytest.raises(ValueError, match="You need to provide either"):
            DummyInputGenerator.pad_input_on_dim(tensor, dim=1, desired_length=15, padding_length=5)

    def test_pad_returns_input_if_no_padding_needed(self):
        """Test pad_input_on_dim returns input if desired_length <= current length."""
        tensor = torch.ones(2, 10)
        result = DummyInputGenerator.pad_input_on_dim(tensor, dim=1, desired_length=5)
        assert result is tensor


class TestTimestepGeneratorAllCases:
    """Test all cases in DummyTimestepInputGenerator."""

    def test_time_ids_input(self):
        """Test generating time_ids."""
        mock_config = MagicMock()
        mock_config.vocab_size = 1000
        mock_config.text_encoder_projection_dim = 1280
        mock_config.time_cond_proj_dim = None
        mock_config.requires_aesthetics_score = False

        gen = DummyTimestepInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        time_ids = gen.generate("time_ids")
        assert time_ids.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 6)

    def test_time_ids_with_aesthetics_score(self):
        """Test generating time_ids with aesthetics score."""
        mock_config = MagicMock()
        mock_config.vocab_size = 1000
        mock_config.text_encoder_projection_dim = 1280
        mock_config.time_cond_proj_dim = None
        mock_config.requires_aesthetics_score = True

        gen = DummyTimestepInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        time_ids = gen.generate("time_ids")
        assert time_ids.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 5)

    def test_timestep_cond_input(self):
        """Test generating timestep_cond."""
        mock_config = MagicMock()
        mock_config.vocab_size = 1000
        mock_config.text_encoder_projection_dim = 1280
        mock_config.time_cond_proj_dim = 256

        gen = DummyTimestepInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        cond = gen.generate("timestep_cond")
        assert cond.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 256)

    def test_unsupported_input_raises(self):
        """Test generating unsupported input raises."""
        mock_config = MagicMock()
        mock_config.vocab_size = 1000
        mock_config.text_encoder_projection_dim = 1280
        mock_config.time_cond_proj_dim = None

        gen = DummyTimestepInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")


class TestDiffusersGeneratorsAllCases:
    """Test all match cases in diffusers generators."""

    def test_sd3_encoder_hidden_states(self):
        """Test SD3 encoder_hidden_states generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 16
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 2048

        gen = DummySD3TransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        hidden = gen.generate("encoder_hidden_states")
        assert hidden.shape == (
            DEFAULT_DUMMY_SHAPES["batch_size"],
            DEFAULT_DUMMY_SHAPES["sequence_length"],
            4096,
        )

    def test_sd3_unsupported_input_raises(self):
        """Test SD3 unsupported input raises."""
        mock_config = MagicMock()
        mock_config.in_channels = 16
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 2048

        gen = DummySD3TransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")

    def test_flux_encoder_hidden_states(self):
        """Test Flux encoder_hidden_states generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 64
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = False

        gen = DummyFluxTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        hidden = gen.generate("encoder_hidden_states")
        assert hidden.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]

    def test_flux_pooled_projections(self):
        """Test Flux pooled_projections generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 64
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = False

        gen = DummyFluxTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        pooled = gen.generate("pooled_projections")
        assert pooled.shape == (DEFAULT_DUMMY_SHAPES["batch_size"], 768)

    def test_flux_timestep(self):
        """Test Flux timestep generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 64
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = False

        gen = DummyFluxTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        timestep = gen.generate("timestep")
        assert timestep.shape == (DEFAULT_DUMMY_SHAPES["batch_size"],)

    def test_flux_unsupported_input_raises(self):
        """Test Flux unsupported input raises."""
        mock_config = MagicMock()
        mock_config.in_channels = 64
        mock_config.sample_size = 128
        mock_config.joint_attention_dim = 4096
        mock_config.pooled_projection_dim = 768
        mock_config.guidance_embeds = False

        gen = DummyFluxTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")

    def test_sana_timestep(self):
        """Test Sana timestep generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 32
        mock_config.sample_size = 32
        mock_config.caption_channels = 2304

        gen = DummySanaTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        timestep = gen.generate("timestep")
        assert timestep.shape == (DEFAULT_DUMMY_SHAPES["batch_size"],)

    def test_sana_encoder_hidden_states(self):
        """Test Sana encoder_hidden_states generation."""
        mock_config = MagicMock()
        mock_config.in_channels = 32
        mock_config.sample_size = 32
        mock_config.caption_channels = 2304

        gen = DummySanaTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        hidden = gen.generate("encoder_hidden_states")
        assert hidden.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]

    def test_sana_unsupported_input_raises(self):
        """Test Sana unsupported input raises."""
        mock_config = MagicMock()
        mock_config.in_channels = 32
        mock_config.sample_size = 32
        mock_config.caption_channels = 2304

        gen = DummySanaTransformerInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")


class TestUNetGeneratorAllCases:
    """Test all cases in DummyUNetInputGenerator."""

    def test_unet_unsupported_input_raises(self):
        """Test UNet unsupported input raises."""
        mock_config = MagicMock()
        mock_config.in_channels = 4
        mock_config.sample_size = 64
        mock_config.cross_attention_dim = 768

        gen = DummyUNetInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")


class TestVaeGeneratorAllCases:
    """Test all cases in DummyVaeInputGenerator."""

    def test_vae_unsupported_input_raises(self):
        """Test VAE unsupported input raises."""
        mock_config = MagicMock()
        mock_config.in_channels = 3
        mock_config.latent_channels = 4
        mock_config.sample_size = 512
        mock_config.down_block_types = ["DownEncoderBlock2D"] * 4

        gen = DummyVaeInputGenerator(TaskType.IMAGE_TO_IMAGE, mock_config)
        with pytest.raises(ValueError, match="Unsupported input name"):
            gen.generate("unsupported_input")


class TestBloomPKVGenerator:
    """Test BloomDummyPastKeyValuesGenerator."""

    def test_bloom_pkv_generator(self):
        """Test BloomDummyPastKeyValuesGenerator."""
        from olive.common.hf.io_config.input_generators import BloomDummyPastKeyValuesGenerator

        mock_config = MagicMock()
        mock_config.num_layers = 24
        mock_config.num_attention_heads = 16
        mock_config.hidden_size = 2048

        gen = BloomDummyPastKeyValuesGenerator(TaskType.TEXT_GENERATION, mock_config)
        pkv = gen.generate("past_key_values")
        assert len(pkv) == 24
        # Bloom uses batch_first format
        key, value = pkv[0]
        assert key.shape[0] == DEFAULT_DUMMY_SHAPES["batch_size"]
