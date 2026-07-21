# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
import torch.nn as nn
from transformers import PretrainedConfig
from transformers.modeling_utils import PreTrainedModel

from olive.common.quant.hf_utils import (
    OliveHfQuantizationConfig,
    OliveHfQuantizationMethod,
    OliveHfQuantizationOverrideConfig,
    OliveHfQuantizer,
    replace_matching_submodules,
    tie_quant_word_embeddings,
)
from olive.common.quant.tensor import QuantTensor

# pylint: disable=W0212


def _is_olive_quant(module: nn.Module) -> bool:
    """Return True if ``module`` is an nn.Linear/nn.Embedding whose weight is a QuantTensor."""
    if not isinstance(module, (nn.Linear, nn.Embedding)):
        return False
    weight = module._parameters.get("weight")
    return weight is not None and isinstance(weight.data, QuantTensor)


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

        # Check that linear layers are replaced with QuantTensor weights
        assert _is_olive_quant(model.linear1)
        assert _is_olive_quant(model.linear2)

        # Check that embeddings and lm_head are NOT quantized by default
        assert not _is_olive_quant(model.embed)
        assert not _is_olive_quant(model.lm_head)

    def test_process_model_with_embeds_quantization(self):
        """Test that embeddings are quantized when embeds=True."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, embeds=True)
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that embeddings are quantized
        assert _is_olive_quant(model.embed)

    def test_process_model_with_lm_head_quantization(self):
        """Test that lm_head is quantized when lm_head=True."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, lm_head=True)
        quantizer = OliveHfQuantizer(config)

        model_config = SimpleConfig()
        model = SimpleModel(model_config)

        # Process model
        quantizer._process_model_before_weight_loading(model)

        # Check that lm_head is quantized
        assert _is_olive_quant(model.lm_head)

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
        assert not _is_olive_quant(model.linear1)
        # But linear2 should be quantized
        assert _is_olive_quant(model.linear2)

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

        # Set different values on the aliased buffers
        model.embed.weight_qweight.fill_(1)
        model.lm_head.weight_qweight.fill_(2)

        # Tie embeddings
        tie_quant_word_embeddings(model)

        # Now they should share the underlying buffers and the Parameter
        assert model.embed.weight_qweight is model.lm_head.weight_qweight
        assert model.embed.weight_scales is model.lm_head.weight_scales
        assert model.embed._parameters["weight"] is model.lm_head._parameters["weight"]


# -- MoE quantization fixtures and tests --------------------------------------


class MoEConfig(PretrainedConfig):
    model_type = "moe_simple"

    def __init__(self, hidden_size=64, vocab_size=64, num_experts=4, num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_experts = num_experts
        self.num_hidden_layers = num_layers
        # required by ModelWrapper / LayerWrapper
        self.num_attention_heads = 4
        self.num_key_value_heads = 4
        self.head_dim = hidden_size // 4
        self.intermediate_size = hidden_size


class _MoEExpert(nn.Module):
    """Per-expert sub-module (ModuleList style — like Mixtral / PhiMoE)."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.w1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.w2 = nn.Linear(hidden_size, hidden_size, bias=False)


class _MoELayer(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int):
        super().__init__()
        self.mlp = nn.Module()
        self.mlp.gate = nn.Linear(hidden_size, num_experts, bias=False)
        self.mlp.experts = nn.ModuleList([_MoEExpert(hidden_size) for _ in range(num_experts)])


class MoESimpleModel(PreTrainedModel):
    config_class = MoEConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = nn.Module()
        self.model.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.model.layers = nn.ModuleList(
            [_MoELayer(config.hidden_size, config.num_experts) for _ in range(config.num_hidden_layers)]
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings


class TestOliveHfQuantizerMoE:
    """MoE-specific behaviour of ``OliveHfQuantizer``.

    The previous implementation silently quantized every per-expert
    ``nn.Linear`` in ``ModuleList(Expert)`` blocks (Mixtral / PhiMoE /
    Qwen2/3-MoE). With the new ``moe`` category flag (default ``False``),
    every ``nn.Module`` under each experts subtree is added to the skip
    set — fixing the silent quantization.
    """

    def test_module_list_experts_skipped_by_default(self):
        """Regression: ModuleList(Expert) linears must stay as nn.Linear when moe=False."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, moe=False)
        quantizer = OliveHfQuantizer(config)
        model = MoESimpleModel(MoEConfig())

        quantizer._process_model_before_weight_loading(model)

        for expert in model.model.layers[0].mlp.experts:
            assert isinstance(expert.w1, nn.Linear)
            assert not _is_olive_quant(expert.w1)
            assert isinstance(expert.w2, nn.Linear)
            assert not _is_olive_quant(expert.w2)
        # The router (gate) is also under the mlp but not under .experts;
        # it should still be quantized.
        assert _is_olive_quant(model.model.layers[0].mlp.gate)

    def test_module_list_experts_quantized_when_moe_true(self):
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16, moe=True)
        quantizer = OliveHfQuantizer(config)
        model = MoESimpleModel(MoEConfig())

        quantizer._process_model_before_weight_loading(model)

        for expert in model.model.layers[0].mlp.experts:
            assert _is_olive_quant(expert.w1)
            assert _is_olive_quant(expert.w2)

    def test_moe_default_is_false(self):
        """``moe`` defaults to False — opt-in, matching lm_head / embeds."""
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=16)
        assert config.moe is False

    def test_regex_skip_pattern(self):
        """``re:`` prefix opts into regex fullmatch for modules_to_not_convert."""
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=16,
            moe=True,
            # Quantize experts but keep the router in fp.
            modules_to_not_convert=["re:.*\\.mlp\\.gate"],
        )
        quantizer = OliveHfQuantizer(config)
        model = MoESimpleModel(MoEConfig())

        quantizer._process_model_before_weight_loading(model)

        # gate is skipped via regex
        assert isinstance(model.model.layers[0].mlp.gate, nn.Linear)
        assert not _is_olive_quant(model.model.layers[0].mlp.gate)
        # experts are quantized
        assert _is_olive_quant(model.model.layers[0].mlp.experts[0].w1)


class TestRegexOverrides:
    def test_get_qlinear_init_args_with_regex_override(self):
        overrides = {
            "re:.*\\.mlp\\.experts\\..*\\.w1": {"bits": 8, "group_size": 32},
            "model.lm_head": {"bits": 4},
        }
        config = OliveHfQuantizationConfig(
            bits=4,
            symmetric=True,
            group_size=128,
            overrides=overrides,
        )
        init_args = config.get_qlinear_init_args("model.layers.0.mlp.experts.0.w1")
        assert init_args == {"bits": 8, "symmetric": True, "group_size": 32}

    def test_first_matching_override_wins_in_insertion_order(self):
        # Two overlapping overrides match the same target. Per the finalized precedence rule
        # (design item 6), the FIRST key in insertion order wins — not the longer/literal one.
        overrides = {
            "re:.*\\.w1": {"bits": 8},
            "model.layers.0.mlp.experts.0.w1": {"bits": 6},
        }
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128, overrides=overrides)
        init_args = config.get_qlinear_init_args("model.layers.0.mlp.experts.0.w1")
        # The regex is first in insertion order, so it wins even though the literal is "more specific".
        assert init_args["bits"] == 8

    def test_reordering_overlapping_overrides_flips_the_winner(self):
        # Same two overlapping overrides, literal placed first -> literal wins.
        overrides = {
            "model.layers.0.mlp.experts.0.w1": {"bits": 6},
            "re:.*\\.w1": {"bits": 8},
        }
        config = OliveHfQuantizationConfig(bits=4, symmetric=True, group_size=128, overrides=overrides)
        init_args = config.get_qlinear_init_args("model.layers.0.mlp.experts.0.w1")
        assert init_args["bits"] == 6
