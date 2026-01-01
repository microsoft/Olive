# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from olive.common.quant.hf_utils import (
    OliveHfQuantizationConfig,
    OliveHfQuantizationMethod,
    OliveHfQuantizationOverrideConfig,
    OliveHfQuantizer,
    replace_matching_submodules,
    tie_quant_modules,
    tie_quant_word_embeddings,
)
from olive.common.quant.nn import QuantEmbedding, QuantLinear

# pylint: disable=W0212


class TestOliveHfQuantizationMethod:
    def test_enum_value(self):
        """Test that the enum has the expected value."""
        assert OliveHfQuantizationMethod.OLIVE == "olive"
        assert OliveHfQuantizationMethod.OLIVE.value == "olive"


class TestOliveHfQuantizationOverrideConfig:
    def test_default_initialization(self):
        """Test default initialization of override config."""
        config = OliveHfQuantizationOverrideConfig()
        assert config.bits is None
        assert config.symmetric is None
        assert config.group_size is None

    def test_custom_initialization(self):
        """Test custom initialization of override config."""
        config = OliveHfQuantizationOverrideConfig(bits=8, symmetric=False, group_size=32)
        assert config.bits == 8
        assert config.symmetric is False
        assert config.group_size == 32

    def test_partial_initialization(self):
        """Test partial initialization of override config."""
        config = OliveHfQuantizationOverrideConfig(bits=4)
        assert config.bits == 4
        assert config.symmetric is None
        assert config.group_size is None


class TestOliveHfQuantizationConfig:
    def test_basic_initialization(self):
        """Test basic initialization with required parameters."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        assert config.bits == 4
        assert config.symmetric is True
        assert config.group_size == 128
        assert config.lm_head is False
        assert config.embeds is False
        assert config.modules_to_not_convert is None
        assert config.overrides == {}
        assert config.quant_method == OliveHfQuantizationMethod.OLIVE

    def test_full_initialization(self):
        """Test initialization with all parameters."""
        overrides = {"layer1": {"bits": 8}, "layer2": {"symmetric": False, "group_size": 64}}
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            lm_head=True,
            embeds=True,
            modules_to_not_convert=["layer3"],
            overrides=overrides,
            tie_word_embeddings=True,
        )
        assert config.bits == 4
        assert config.symmetric is True
        assert config.group_size == 128
        assert config.lm_head is True
        assert config.embeds is True
        assert config.modules_to_not_convert == ["layer3"]
        assert len(config.overrides) == 2
        assert isinstance(config.overrides["layer1"], OliveHfQuantizationOverrideConfig)
        assert config.overrides["layer1"].bits == 8
        assert config.tie_word_embeddings is True

    def test_invalid_bits(self):
        """Test that invalid bits raise ValueError."""
        with pytest.raises(ValueError, match="Only 2-bit, 4-bit and 8-bit quantization supported"):
            OliveHfQuantizationConfig(bits=16, symmetric=True, group_size=128)

    def test_to_dict(self):
        """Test serialization to dictionary."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        config_dict = config.to_dict()
        assert config_dict["bits"] == 4
        assert config_dict["symmetric"] is True
        assert config_dict["group_size"] == 128
        assert config_dict["quant_method"] == "olive"

    def test_to_dict_with_overrides(self):
        """Test serialization with overrides."""
        overrides = {"layer1": {"bits": 8}}
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            overrides=overrides,
        )
        config_dict = config.to_dict()
        assert "overrides" in config_dict
        assert config_dict["overrides"]["layer1"]["bits"] == 8

    def test_to_dict_removes_default_overrides(self):
        """Test that to_dict removes override values that match defaults."""
        overrides = {"layer1": {"bits": 4}}  # Same as default bits
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            overrides=overrides,
        )
        config_dict = config.to_dict()
        # The override should be removed or empty since it matches the default
        assert config_dict.get("overrides") is None or "layer1" not in config_dict.get("overrides", {})

    def test_get_qlinear_init_args_no_override(self):
        """Test getting initialization args without overrides."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        init_args = config.get_qlinear_init_args("some_layer")
        assert init_args == {"bits": 4, "symmetric": True, "group_size": 128}

    def test_get_qlinear_init_args_with_override(self):
        """Test getting initialization args with overrides."""
        overrides = {"layer1": {"bits": 8, "group_size": 64}}
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            overrides=overrides,
        )
        init_args = config.get_qlinear_init_args("layer1")
        assert init_args == {"bits": 8, "symmetric": True, "group_size": 64}

    def test_get_qlinear_init_args_with_partial_override(self):
        """Test getting initialization args with partial overrides."""
        overrides = {"layer1": {"bits": 8}}
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            overrides=overrides,
        )
        init_args = config.get_qlinear_init_args("layer1")
        assert init_args == {"bits": 8, "symmetric": True, "group_size": 128}


# Simple test model for OliveHfQuantizer tests
class SimpleConfig(PretrainedConfig):
    model_type = "simple"

    def __init__(self, hidden_size=64, vocab_size=100, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size


class SimpleModel(PreTrainedModel):
    config_class = SimpleConfig

    def __init__(self, config):
        super().__init__(config)
        self.embed = nn.Embedding(config.vocab_size, config.hidden_size)
        self.linear1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.linear2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class TestOliveHfQuantizer:
    def test_quantizer_initialization(self):
        """Test that quantizer initializes correctly."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        quantizer = OliveHfQuantizer(config)
        assert quantizer.quantization_config == config
        assert quantizer.requires_calibration is True

    def test_process_model_before_weight_loading(self):
        """Test that model is correctly modified before weight loading."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16)
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that linear layers are replaced with QuantLinear
        assert isinstance(model.linear1, QuantLinear)
        assert isinstance(model.linear2, QuantLinear)

        # Check that embeddings and lm_head are NOT quantized by default
        assert not isinstance(model.embed, QuantEmbedding)
        assert not isinstance(model.lm_head, QuantLinear)

    def test_process_model_with_embeds_quantization(self):
        """Test that embeddings are quantized when embeds=True."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, embeds=True)
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that embeddings are quantized
        assert isinstance(model.embed, QuantEmbedding)

    def test_process_model_with_lm_head_quantization(self):
        """Test that lm_head is quantized when lm_head=True."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, lm_head=True)
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that lm_head is quantized
        assert isinstance(model.lm_head, QuantLinear)

    def test_process_model_with_modules_to_not_convert(self):
        """Test that specified modules are not converted."""
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=16,
            modules_to_not_convert=["linear1"],
        )
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that linear1 is NOT quantized
        assert isinstance(model.linear1, nn.Linear)
        # But linear2 should be quantized
        assert isinstance(model.linear2, QuantLinear)

    def test_is_serializable(self):
        """Test that quantizer is serializable."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        quantizer = OliveHfQuantizer(config)
        assert quantizer.is_serializable() is True

    def test_is_trainable(self):
        """Test that quantizer reports not trainable."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128)
        quantizer = OliveHfQuantizer(config)
        assert quantizer.is_trainable is False


class TestReplaceMatchingSubmodules:
    def test_replace_all_linear_layers(self):
        """Test replacing all linear layers in a module."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(10, 20)
                self.linear2 = nn.Linear(20, 30)
                self.relu = nn.ReLU()

        module = SimpleModule()

        def condition(m, name):
            return isinstance(m, nn.Linear)

        def transform(m, name):
            return nn.Identity()

        result = replace_matching_submodules(module, condition, transform, description="Test replacement")

        # Check that linear layers are replaced
        assert isinstance(result.linear1, nn.Identity)
        assert isinstance(result.linear2, nn.Identity)
        # ReLU should remain unchanged
        assert isinstance(result.relu, nn.ReLU)

    def test_replace_with_name_filter(self):
        """Test replacing modules based on name."""

        class SimpleModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.keep_linear = nn.Linear(10, 20)
                self.replace_linear = nn.Linear(20, 30)

        module = SimpleModule()

        def condition(m, name):
            return isinstance(m, nn.Linear) and "replace" in name

        def transform(m, name):
            return nn.Identity()

        result = replace_matching_submodules(module, condition, transform)

        # Only replace_linear should be replaced
        assert isinstance(result.keep_linear, nn.Linear)
        assert isinstance(result.replace_linear, nn.Identity)

    def test_replace_nested_modules(self):
        """Test replacing modules in nested structure."""

        class NestedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.sub = nn.Sequential(
                    nn.Linear(10, 20),
                    nn.ReLU(),
                    nn.Linear(20, 30),
                )

        module = NestedModule()

        def condition(m, name):
            return isinstance(m, nn.Linear)

        def transform(m, name):
            return nn.Identity()

        result = replace_matching_submodules(module, condition, transform)

        # Check nested replacements
        assert isinstance(result.sub[0], nn.Identity)
        assert isinstance(result.sub[1], nn.ReLU)
        assert isinstance(result.sub[2], nn.Identity)


class TestTieQuantModules:
    def test_tie_quant_linear_modules(self):
        """Test tying two QuantLinear modules."""
        # Create two QuantLinear modules
        qlinear1 = QuantLinear(32, 10, bits=4, symmetric=True, group_size=16)
        qlinear2 = QuantLinear(32, 10, bits=4, symmetric=True, group_size=16)

        # Set some values in qlinear1
        qlinear1.qweight.fill_(1)
        qlinear1.scales.fill_(0.5)

        # Initially different
        assert not torch.all(qlinear1.qweight == qlinear2.qweight)

        # Tie them
        tie_quant_modules(qlinear1, qlinear2)

        # Now they should share buffers
        assert qlinear1.qweight is qlinear2.qweight
        assert qlinear1.scales is qlinear2.scales

        # Modifications to one should affect the other
        qlinear1.qweight.fill_(2)
        assert torch.all(qlinear2.qweight == 2)

    def test_tie_quant_embedding_modules(self):
        """Test tying two QuantEmbedding modules."""
        # Create two QuantEmbedding modules
        qembed1 = QuantEmbedding(100, 64, bits=4, symmetric=True, group_size=16)
        qembed2 = QuantEmbedding(100, 64, bits=4, symmetric=True, group_size=16)

        # Set some values in qembed1
        qembed1.qweight.fill_(1)
        qembed1.scales.fill_(0.5)

        # Tie them
        tie_quant_modules(qembed1, qembed2)

        # Now they should share buffers
        assert qembed1.qweight is qembed2.qweight
        assert qembed1.scales is qembed2.scales


class TestTieQuantWordEmbeddings:
    def test_tie_word_embeddings(self):
        """Test tying word embeddings in a model."""
        # Create model with quantized embeddings
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=16,
            embeds=True,
            lm_head=True,
        )
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model to quantize embeddings
        quantizer._process_model_before_weight_loading(model)

        # Set different values
        model.embed.qweight.fill_(1)
        model.lm_head.qweight.fill_(2)

        # Tie embeddings
        tie_quant_word_embeddings(model)

        # Now they should share buffers
        assert model.embed.qweight is model.lm_head.qweight
        assert model.embed.scales is model.lm_head.scales
