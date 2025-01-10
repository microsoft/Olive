# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

from typing import Iterable

import torch
from torch import nn

from olive.common.hf.adapter import ModelAdapter
from olive.common.utils import cleanup_memory
from olive.passes.pytorch.hadamard_utils import random_hadamard_matrix

# ruff: noqa: N806


def fuse_ln_linear(layernorm: nn.Module, linear_layers: Iterable[nn.Linear]):
    """Fuse the linear operations in Layernorm into the adjacent linear blocks."""
    for linear in linear_layers:
        linear_dtype = linear.weight.dtype

        # Calculating new weight and bias
        W_ = linear.weight.data.double()
        linear.weight.data = (W_ * layernorm.weight.double()).to(linear_dtype)

        if hasattr(layernorm, "bias"):
            if linear.bias is None:
                linear.bias = nn.Parameter(torch.zeros(linear.out_features, dtype=torch.float64))
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


def rotate_embed(embedding: nn.Embedding, Q: torch.Tensor, device: torch.device):
    """Rotate the embedding matrix by Q."""
    w_device = embedding.weight.data.device
    w_dtype = embedding.weight.dtype

    W = embedding.weight.data.to(device=device, dtype=torch.float64)
    embedding.weight.data = torch.matmul(W, Q).to(device=w_device, dtype=w_dtype)


def rotate_pre_linear(linear: nn.Linear, Q: torch.Tensor, device: torch.device):
    """Rotate the input linear layer by Q."""
    w_device = linear.weight.device
    w_dtype = linear.weight.data.dtype

    out_features, in_features = linear.weight.data.shape
    q_features = Q.shape[0]

    headwise = False
    if in_features != q_features:
        assert in_features % q_features == 0, "Input features should be divisible by Q features"
        headwise = True

    W = linear.weight.data.to(device=device, dtype=torch.float64)
    if headwise:
        W = W.reshape(out_features, -1, q_features)
        linear.weight.data = torch.matmul(W, Q).reshape(out_features, -1).to(device=w_device, dtype=w_dtype)
    else:
        linear.weight.data = torch.matmul(W, Q).to(device=w_device, dtype=w_dtype)


def rotate_post_linear(linear: nn.Linear, Q: torch.Tensor, device: torch.device):
    """Rotate the output linear layer by Q."""
    w_device = linear.weight.device
    w_dtype = linear.weight.data.dtype

    out_features, in_features = linear.weight.data.shape
    q_features = Q.shape[0]

    headwise = False
    if out_features != q_features:
        assert out_features % q_features == 0, "Output features should be divisible by Q features"
        headwise = True

    W = linear.weight.data.to(device=device, dtype=torch.float64)
    if headwise:
        W = W.t().reshape(in_features, -1, q_features)
        linear.weight.data = torch.matmul(W, Q).reshape(in_features, -1).t().to(device=w_device, dtype=w_dtype)
    else:
        linear.weight.data = torch.matmul(Q.T, W).to(device=w_device, dtype=w_dtype)

    if hasattr(linear, "bias") and linear.bias is not None:
        b = linear.bias.data.to(device=device, dtype=torch.float64)
        if headwise:
            b = b.reshape(-1, q_features)
            linear.bias.data = torch.matmul(b, Q).reshape(-1).to(device=w_device, dtype=w_dtype)
        else:
            linear.bias.data = torch.matmul(Q.T, b).to(device=w_device, dtype=w_dtype)


def random_orthogonal_matrix(size: int, device: torch.device) -> torch.Tensor:
    """Generate a random orthogonal matrix of the specified size.

    First, we generate a random matrix with entries from a standard distribution.
    Then, we use QR decomposition to obtain an orthogonal matrix.
    Finally, we multiply by a diagonal matrix with diag r to adjust the signs.
    """
    cleanup_memory()
    random_matrix = torch.randn(size, size, dtype=torch.float64).to(device)
    q, r = torch.linalg.qr(random_matrix)  # pylint: disable=not-callable
    q *= torch.sign(torch.diag(r)).unsqueeze(0)
    return q


def get_orthogonal_matrix(size: int, mode: str, device: torch.device) -> torch.Tensor:
    """Get an orthogonal matrix of the specified size.

    Supported modes:
    - random: generate a random orthogonal matrix
    - hadamard: generate a random Hadamard matrix
    """
    if mode == "random":
        return random_orthogonal_matrix(size, device)
    elif mode == "hadamard":
        return random_hadamard_matrix(size, device)
    else:
        raise ValueError(f"Unknown mode {mode}")
