# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from torch import nn
from transformers import PretrainedConfig

from olive.common.hf.wrapper import ModelWrapper
from test.utils import get_tiny_phi3, make_local_tiny_llama


@pytest.mark.parametrize("model_path", ["tiny-phi3", "tiny-llama"])
def test_hf_wrapper(model_path, tmp_path):
    if model_path == "tiny-llama":
        input_model = make_local_tiny_llama(tmp_path / "model")
    else:
        input_model = get_tiny_phi3()

    model_wrapper = ModelWrapper(input_model.get_hf_model_config())

    # check for the model attributes
    for key in ["model_type", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "head_dim"]:
        assert getattr(model_wrapper, key) is not None

    # model has not been loaded yet
    with pytest.raises(ValueError, match=r"Model is not set\."):
        _ = model_wrapper.model

    # load the model
    loaded_model = input_model.load_model()
    model_wrapper.set_model(loaded_model)
    assert model_wrapper.model is loaded_model

    # check the high-level submodules
    assert isinstance(model_wrapper.get_embeds(False)[0], nn.Embedding)
    assert isinstance(model_wrapper.get_lm_head(False), nn.Linear)
    assert model_wrapper.get_pre_head_layernorm(False).__class__.__name__.endswith("RMSNorm")
    assert len(model_wrapper.get_layers(False)) == model_wrapper.num_hidden_layers

    # get the first layer adapter
    layer_wrapper = model_wrapper.get_layer_wrappers()[0]

    # layernorms in the layer block
    for key in ["get_first_layer_norm", "get_second_layer_norm"]:
        assert getattr(layer_wrapper, key)(False).__class__.__name__.endswith("RMSNorm")

    # projection layers in the layer block
    for key in ["get_attention_inputs", "get_attention_outputs", "get_mlp_inputs", "get_mlp_outputs"]:
        modules, names = getattr(layer_wrapper, key)()
        for module in modules:
            assert isinstance(module, nn.Linear)
        for name in names:
            assert name.startswith(("self_attn.", "mlp."))

    if model_wrapper.model_type == "llama":
        # qkv is already split
        assert len(layer_wrapper.get_attention_inputs(False)) == 3
    elif model_wrapper.model_type == "phi3":
        # qkv is a single linear layer
        assert len(layer_wrapper.get_attention_inputs(False)) == 1

        # split qkv
        model_wrapper.maybe_unpack_qkv()

        # check the split qkv
        modules, names = layer_wrapper.get_attention_inputs()
        assert len(modules) == 3
        for module in modules:
            assert isinstance(module, nn.Linear)
        for name in names:
            assert name.startswith("self_attn.qkv_proj")


def test_hf_wrapper_lfm2():
    """Test LayerWrapper with LFM2 hybrid model (conv + attention layers)."""
    from olive.model import HfModelHandler

    input_model = HfModelHandler(model_path="tiny-random/lfm2")
    model_wrapper = ModelWrapper(input_model.get_hf_model_config())

    assert model_wrapper.model_type == "lfm2"

    loaded_model = input_model.load_model()
    model_wrapper.set_model(loaded_model)

    # high-level submodules
    assert isinstance(model_wrapper.get_embeds(False)[0], nn.Embedding)
    assert isinstance(model_wrapper.get_lm_head(False), nn.Linear)
    assert model_wrapper.get_pre_head_layernorm(False).__class__.__name__.endswith("RMSNorm")

    layer_wrappers = model_wrapper.get_layer_wrappers()
    assert len(layer_wrappers) == model_wrapper.num_hidden_layers

    has_attn_layer = False
    has_conv_layer = False

    for layer_wrapper in layer_wrappers:
        # all layers have layernorms and MLP
        assert layer_wrapper.get_first_layer_norm(False).__class__.__name__.endswith("RMSNorm")
        assert layer_wrapper.get_second_layer_norm(False).__class__.__name__.endswith("RMSNorm")

        mlp_modules, mlp_names = layer_wrapper.get_mlp_inputs()
        assert len(mlp_modules) == 2
        for m in mlp_modules:
            assert isinstance(m, nn.Linear)
        for n in mlp_names:
            assert n.startswith("feed_forward.")

        mlp_out_modules, mlp_out_names = layer_wrapper.get_mlp_outputs()
        assert len(mlp_out_modules) == 1
        assert isinstance(mlp_out_modules[0], nn.Linear)
        assert mlp_out_names[0].startswith("feed_forward.")

        if layer_wrapper.attn is not None:
            # attention layer
            has_attn_layer = True
            attn_modules, attn_names = layer_wrapper.get_attention_inputs()
            assert len(attn_modules) == 3
            for m in attn_modules:
                assert isinstance(m, nn.Linear)

            attn_out_modules, attn_out_names = layer_wrapper.get_attention_outputs()
            assert len(attn_out_modules) == 1
            assert isinstance(attn_out_modules[0], nn.Linear)
            assert attn_out_names[0].startswith("self_attn.")
        else:
            # conv layer — attention methods return empty
            has_conv_layer = True
            attn_modules, attn_names = layer_wrapper.get_attention_inputs()
            assert attn_modules == []
            assert attn_names == []

    # LFM2 must have both layer types
    assert has_attn_layer, "Expected at least one attention layer"
    assert has_conv_layer, "Expected at least one conv layer"


def test_hf_wrapper_composite_vl_config():
    """Composite VL configs (e.g. Qwen3.5/3.6-MoE VL) nest decoder attributes under ``text_config``.

    Verifies the ``text_config.*`` alias fallbacks in ``defaults.yaml`` resolve them, and the
    ``qwen3_5_moe`` entries in ``LAYERS``/``EMBEDDINGS``/``PRE_HEAD_LAYERNORM``/
    ``ROTARY_EMBEDDING`` point at the VL decoder's actual nested module path
    (``model.language_model.*``), without needing to download/load any real model or weights.

    ``text_config``/``vision_config`` must be actual (attribute-accessible) nested
    ``PretrainedConfig`` objects here -- as real composite VL config classes (e.g.
    ``Qwen3_5MoeConfig``) construct them -- not plain dicts, since ``resolve_alias``'s nested
    path lookup uses ``getattr``, not dict indexing.
    """
    text_config = PretrainedConfig()
    text_config.hidden_size = 2048
    text_config.num_hidden_layers = 40
    text_config.num_attention_heads = 16
    text_config.num_key_value_heads = 2
    text_config.head_dim = 256
    text_config.num_experts = 256

    vision_config = PretrainedConfig()
    vision_config.hidden_size = 1152

    composite_config = PretrainedConfig()
    composite_config.model_type = "qwen3_5_moe"
    composite_config.text_config = text_config
    composite_config.vision_config = vision_config

    model_wrapper = ModelWrapper(composite_config)

    assert model_wrapper.model_type == "qwen3_5_moe"
    assert model_wrapper.hidden_size == 2048
    assert model_wrapper.num_attention_heads == 16
    assert model_wrapper.num_key_value_heads == 2
    assert model_wrapper.head_dim == 256
    assert model_wrapper.num_hidden_layers == 40

    assert model_wrapper.LAYERS[model_wrapper.model_type] == "model.language_model.layers"
    assert model_wrapper.EMBEDDINGS[model_wrapper.model_type] == ["model.language_model.embed_tokens"]
    assert model_wrapper.PRE_HEAD_LAYERNORM[model_wrapper.model_type] == "model.language_model.norm"
    assert model_wrapper.ROTARY_EMBEDDING[model_wrapper.model_type] == "model.language_model.rotary_emb"


def test_hf_wrapper_text_only_config_unaffected_by_vl_aliases():
    """Verify the flat text-only Qwen3.5/3.6 config resolves exactly as before.

    ``model_type == "qwen3_5_moe_text"`` already has flat attributes matching the "default"
    aliases, and has no ``qwen3_5_moe`` entry in the ``LAYERS``/etc. mappings (a different
    model_type), so the VL-specific additions must not change its resolution.
    """
    text_only_config = {
        "model_type": "qwen3_5_moe_text",
        "hidden_size": 2048,
        "num_hidden_layers": 40,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "num_experts": 256,
    }

    model_wrapper = ModelWrapper(text_only_config)

    assert model_wrapper.model_type == "qwen3_5_moe_text"
    assert model_wrapper.hidden_size == 2048
    assert model_wrapper.num_hidden_layers == 40
    # No "qwen3_5_moe_text" entry in LAYERS/EMBEDDINGS/etc. -- falls back to "default".
    assert model_wrapper.LAYERS.get(model_wrapper.model_type, model_wrapper.LAYERS["default"]) == "model.layers"


def test_hf_wrapper_flat_text_only_qwen3_5_moe_config():
    """Regression: flat text-only checkpoints also report ``model_type == "qwen3_5_moe"``.

    Only the composite VL checkpoint nests its decoder under ``model.language_model``; a
    text-only ``Qwen3_5MoeForCausalLM`` checkpoint keeps the flat ``model.layers`` layout
    while still reporting ``model_type == "qwen3_5_moe"``. The two are disambiguated by the
    presence of a ``vision_config`` (same rule mobius uses), so the flat config must resolve
    the "default" module paths instead of crashing on the VL-only paths.
    """

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.mlp = nn.Module()

    class _FlatQwenMoE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.embed_tokens = nn.Embedding(16, 8)
            self.model.layers = nn.ModuleList([_Layer(), _Layer()])
            self.model.norm = nn.LayerNorm(8)
            self.lm_head = nn.Linear(8, 16, bias=False)

    flat_config = PretrainedConfig()
    flat_config.model_type = "qwen3_5_moe"
    flat_config.hidden_size = 2048
    flat_config.num_hidden_layers = 2
    flat_config.num_attention_heads = 16
    flat_config.num_key_value_heads = 2
    flat_config.head_dim = 256
    flat_config.num_experts = 256

    model_wrapper = ModelWrapper(flat_config)

    # normalized to the text-only model_type, so the flat "default" paths are used
    assert model_wrapper.model_type == "qwen3_5_moe_text"
    assert model_wrapper.hidden_size == 2048
    assert model_wrapper.head_dim == 256

    model = _FlatQwenMoE(flat_config)
    model_wrapper.set_model(model)

    assert model_wrapper.get_layers(False) is model.model.layers
    assert model_wrapper.get_embeds(False)[0] is model.model.embed_tokens
    assert model_wrapper.get_pre_head_layernorm(False) is model.model.norm
    assert model_wrapper.get_lm_head(False) is model.lm_head


def test_hf_wrapper_vl_qwen3_5_moe_config_uses_nested_paths():
    """The composite VL config (has ``vision_config``) keeps the nested decoder paths."""

    class _Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = nn.Module()
            self.mlp = nn.Module()

    class _VLQwenMoE(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config
            self.model = nn.Module()
            self.model.language_model = nn.Module()
            self.model.language_model.embed_tokens = nn.Embedding(16, 8)
            self.model.language_model.layers = nn.ModuleList([_Layer()])
            self.model.language_model.norm = nn.LayerNorm(8)
            self.model.visual = nn.Module()
            self.model.visual.proj = nn.Linear(8, 8, bias=False)
            self.lm_head = nn.Linear(8, 16, bias=False)

    text_config = PretrainedConfig()
    text_config.hidden_size = 2048
    text_config.num_hidden_layers = 1
    text_config.num_attention_heads = 16
    text_config.num_key_value_heads = 2
    text_config.head_dim = 256

    vl_config = PretrainedConfig()
    vl_config.model_type = "qwen3_5_moe"
    vl_config.text_config = text_config
    vl_config.vision_config = PretrainedConfig()

    model_wrapper = ModelWrapper(vl_config)
    assert model_wrapper.model_type == "qwen3_5_moe"

    model = _VLQwenMoE(vl_config)
    model_wrapper.set_model(model)

    assert model_wrapper.get_layers(False) is model.model.language_model.layers
    assert model_wrapper.get_embeds(False)[0] is model.model.language_model.embed_tokens
    assert model_wrapper.get_pre_head_layernorm(False) is model.model.language_model.norm
