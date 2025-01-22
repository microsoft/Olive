# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from torch import nn
from transformers import AutoConfig, AutoModelForCausalLM

from olive.common.hf.wrapper import ModelWrapper


@pytest.mark.parametrize(
    "model_path", ["katuni4ka/tiny-random-phi3", "hf-internal-testing/tiny-random-LlamaForCausalLM"]
)
def test_hf_wrapper(model_path):
    model_wrapper = ModelWrapper(AutoConfig.from_pretrained(model_path))

    # check for the model attributes
    for key in ["model_type", "hidden_size", "num_attention_heads", "num_key_value_heads", "head_dim", "head_dim"]:
        assert getattr(model_wrapper, key) is not None

    # model has not been loaded yet
    with pytest.raises(ValueError, match="Model is not set."):
        _ = model_wrapper.model

    # load the model
    loaded_model = AutoModelForCausalLM.from_pretrained(model_path)
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
