# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# Based on QuaRot(https://github.com/spcl/QuaRot/tree/main/quarot).
# Licensed under Apache License 2.0.
import logging
from typing import Iterable, Optional, Union

import torch
from torch import nn

from olive.common.hf.adapter import ModelAdapter
from olive.common.utils import cleanup_memory
from olive.passes.pytorch.utils.hadamard import random_hadamard_matrix

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
    if isinstance(model_adapter.get_pre_head_layernorm(False), nn.LayerNorm):
        raise ValueError("Model uses LayerNorm. Only RMSNorm fusion is supported.")

    logger.debug("Fusing layernorms into adjacent linear layers")

    # untie embedding and lm head
    model_adapter.maybe_untie_word_embeddings()

    # Layers: Fuse layernorms into adjacent linear layers
    for layer_adapter in model_adapter.get_layer_adapters(False):
        fuse_ln_linear(layer_adapter.get_first_layer_norm(False), layer_adapter.get_attention_inputs(False))
        fuse_ln_linear(layer_adapter.get_second_layer_norm(False), layer_adapter.get_mlp_inputs(False))

    # LM Head: Fuse layernorm into linear layer
    fuse_ln_linear(model_adapter.get_pre_head_layernorm(False), [model_adapter.get_lm_head(False)])


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


class RotateEmbed(nn.Module):
    """Embedding layer with pre-rotation."""

    def __init__(self, embedding: nn.Embedding, Q: nn.Parameter):
        super().__init__()
        self.embedding = embedding
        self.Q = Q

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (torch.matmul(self.embedding(x).to(torch.float64), self.Q.to(torch.float64))).to(
            self.embedding.weight.dtype
        )

    @torch.no_grad()
    def create_merged(self, device) -> nn.Embedding:
        """Create a merged embedding layer with the rotation matrix."""
        embed = nn.Embedding.from_pretrained(
            rotate_weight(self.embedding.weight.data, self.Q, pre=True, device=device).contiguous(),
        )
        cleanup_memory()
        return embed


class RotateLinear(nn.Module):
    """Linear layer with pre/post rotations."""

    def __init__(self, linear: nn.Linear, Q_pre: Optional[nn.Parameter] = None, Q_post: Optional[nn.Parameter] = None):
        super().__init__()
        self.linear = linear
        self.Q_pre = Q_pre
        self.Q_post = Q_post

    def get_rotated_weights(self, device: Optional[torch.device] = None) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        weight = self.linear.weight
        bias = self.linear.bias if hasattr(self.linear, "bias") and self.linear.bias is not None else None

        if self.Q_pre is not None:
            weight = rotate_weight(weight, self.Q_pre, pre=True, device=device)
        if self.Q_post is not None:
            weight = rotate_weight(weight, self.Q_post, pre=False, device=device)
            if bias is not None:
                bias = rotate_weight(bias.unsqueeze(1), self.Q_post, pre=False, device=device).squeeze(1)

        return weight, bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.linear(x, *self.get_rotated_weights())  # pylint: disable=not-callable

    @torch.no_grad()
    def create_merged(self, device) -> nn.Linear:
        """Create a merged linear layer with the rotation matrices."""
        weight, bias = self.get_rotated_weights(device)

        linear = nn.Linear(weight.size(1), weight.size(0), bias is not None)
        linear.weight = nn.Parameter(weight.contiguous())
        if bias is not None:
            linear.bias = nn.Parameter(bias.contiguous())

        return linear
