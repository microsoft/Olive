# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This code is based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.

import logging
from typing import Iterable, Optional, Union

import torch
from torch import nn

from olive.common.hf.adapter import ModelAdapter
from olive.common.utils import cleanup_memory
from olive.passes.pytorch.hadamard_utils import random_hadamard_matrix

logger = logging.getLogger(__name__)

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
    if isinstance(model_adapter.get_pre_head_layernorm(), nn.LayerNorm):
        raise ValueError("Model uses LayerNorm. Only RMSNorm fusion is supported.")

    logger.debug("Fusing layernorms into adjacent linear layers")

    # untie embedding and lm head
    model_adapter.maybe_untie_word_embeddings()

    # Layers: Fuse layernorms into adjacent linear layers
    for layer_adapter in model_adapter.get_layer_adapters():
        fuse_ln_linear(layer_adapter.get_first_layer_norm(), layer_adapter.get_attention_inputs())
        fuse_ln_linear(layer_adapter.get_second_layer_norm(), layer_adapter.get_mlp_inputs())

    # LM Head: Fuse layernorm into linear layer
    fuse_ln_linear(model_adapter.get_pre_head_layernorm(), [model_adapter.get_lm_head()])


def rotate_weight(
    weight: Union[torch.Tensor, nn.Parameter],
    Q: Union[torch.Tensor, nn.Parameter],
    pre: bool = True,
    device: Optional[torch.device] = None,
) -> Union[torch.Tensor, nn.Parameter]:
    """Rotate the weight matrix by Q.

    :param weight: weight matrix. Shape is (out_features, in_features). Equivalent to W^T in y = xW + b
    :param Q: rotation matrix. If the Q dimension is different from the weight dimension, head-wise rotation
        is performed.
    :pre: whether to apply the rotation before the linear operation.
        True for pre-rotation: y' = (xQ^-1)W + b = x(Q^TW) + b
        False for post-rotation: y' = yQ = x(WQ) + bQ
    :param device: device to use for the computation.
    """
    dtype = weight.dtype
    original_device = weight.device
    out_features, in_features = weight.shape
    q_features = Q.shape[0]

    head_wise = (in_features != q_features) if pre else (out_features != q_features)

    # Convert to double precision, optionally move to device
    to_kwargs = {"dtype": torch.float64}
    if device is not None:
        to_kwargs["device"] = device
    weight = weight.to(**to_kwargs)
    Q = Q.to(**to_kwargs)

    if pre:
        # Q^T @ W = (W^T @ Q)^T = (weight @ Q)^T
        if head_wise:
            weight = weight.reshape(out_features, -1, q_features)
            weight = torch.matmul(weight, Q).reshape(out_features, -1)
        else:
            weight = torch.matmul(weight, Q)
    else:
        # W @ Q = (Q^T @ W^T)^T = (Q^T @ weight)^T
        if head_wise:
            weight = weight.t().reshape(in_features, -1, q_features)
            weight = torch.matmul(weight, Q).reshape(in_features, -1).t()
        else:
            weight = torch.matmul(Q.t(), weight)

    # Convert back to original precision and device
    to_kwargs = {"dtype": dtype}
    if device is not None:
        to_kwargs["device"] = original_device
    return weight.to(**to_kwargs)


def rotate_embed(embedding: nn.Embedding, Q: torch.Tensor, device: torch.device):
    """Rotate the embedding matrix by Q."""
    embedding.weight.data = rotate_weight(embedding.weight.data, Q, pre=True, device=device)


def rotate_pre_linear(linear: nn.Linear, Q: torch.Tensor, device: torch.device):
    """Rotate the input linear layer by Q."""
    linear.weight.data = rotate_weight(linear.weight.data, Q, pre=True, device=device)


def rotate_post_linear(linear: nn.Linear, Q: torch.Tensor, device: torch.device):
    """Rotate the output linear layer by Q."""
    # transpose and reshape during headwise rotation might make the weight matrix non-contiguous
    linear.weight.data = rotate_weight(linear.weight.data, Q, pre=False, device=device).contiguous()
    if hasattr(linear, "bias") and linear.bias is not None:
        linear.bias.data = (
            rotate_weight(linear.bias.data.unsqueeze(1), Q, pre=False, device=device).squeeze(1).contiguous()
        )


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
