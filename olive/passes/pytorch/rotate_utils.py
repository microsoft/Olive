# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This code is based on TransformerCompression (https://github.com/microsoft/TransformerCompression)

from typing import Iterable

import torch
from torch.nn import Linear, Module, Parameter

from olive.common.hf.adapter import ModelAdapter

# ruff: noqa: N806


def fuse_ln_linear(layernorm: Module, linear_layers: Iterable[Linear]):
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
            linear.bias.data = linear.bias.data.double() + torch.matmul(W_, layernorm.bias.double())
            linear.bias.data = linear.bias.data.to(linear_dtype)

    # update layernorm weight and bias
    layernorm.weight.data = torch.ones_like(layernorm.weight.data)
    if hasattr(layernorm, "bias"):
        layernorm.bias = None


def fuse_layer_norms(model_adapter: ModelAdapter):
    """Fuse layernorms into adjacent linear layers."""
    # TODO(jambayk): should we support models with layernorms? these require:
    # - subtracting mean from embedding
    # - baking mean into output layers
    # - replacing layernorm with RMSNorm
    # Model architecture changes are required

    # untie embedding and lm head
    model_adapter.maybe_untie_word_embeddings()

    # Layers: Fuse layernorms into adjacent linear layers
    for layer_adapter in model_adapter.get_layer_adapters():
        fuse_ln_linear(layer_adapter.get_first_layer_norm(), layer_adapter.get_attention_inputs())
        fuse_ln_linear(layer_adapter.get_second_layer_norm(), layer_adapter.get_mlp_inputs())

    # LM Head: Fuse layernorm into linear layer
    fuse_ln_linear(model_adapter.get_pre_head_layernorm(), [model_adapter.get_lm_head()])
