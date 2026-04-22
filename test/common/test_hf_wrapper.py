# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from torch import nn

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
    with pytest.raises(ValueError, match="Model is not set."):
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

            attn_out_modules, attn_out_names = layer_wrapper.get_attention_outputs()
            assert attn_out_modules == []
            assert attn_out_names == []

    # LFM2 must have both layer types
    assert has_attn_layer, "Expected at least one attention layer"
    assert has_conv_layer, "Expected at least one conv layer"
