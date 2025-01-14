# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# This code is based on SpinQuant(https://github.com/facebookresearch/SpinQuant).
# Licensed under Apache License 2.0.
import torch
from torch import nn


class QuantizeDequantizeSTEFunction(torch.autograd.Function):
    """Quantize-Dequantize function with Straight-Through Estimator."""

    # pylint: disable=W0223,W0221
    @staticmethod
    def forward(ctx, x, scale, zero, maxq):
        q = torch.clamp(torch.round(x / scale) + zero, 0, maxq)
        return scale * (q - zero)

    @staticmethod
    def backward(ctx, grad_q):
        return grad_q, None, None, None


class ActQuantLinear(nn.Module):
    """Linear Module with quantized activations."""

    def __init__(self, linear: nn.Module, bits: int, symmetric: bool = True, per_token: bool = True):
        super().__init__()
        self.linear = linear
        self.bits = bits
        self.symmetric = symmetric
        self.per_token = per_token

        self.maxq = torch.tensor(2**bits - 1)

    def get_qparams(self, x: torch.Tensor):
        device = x.device

        # put maxq on the same device as x
        self.maxq = self.maxq.to(device)

        x_shape = x.shape
        if self.per_token:
            x = x.view(-1, x_shape[-1])
        else:
            x = x.flatten().unsqueeze(0)

        # range needs to include 0
        tmp = torch.zeros(x.shape[0], device=device)
        xmin = torch.minimum(x.min(1)[0], tmp)
        xmax = torch.maximum(x.max(1)[0], tmp)

        if self.symmetric:
            # symmetric quantization has same range on both sides of 0
            tmp = torch.maximum(torch.abs(xmin), xmax)
            xmin = -tmp
            xmax = tmp

        # avoid zero scale
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = 1

        scale = (xmax - xmin) / self.maxq
        if self.symmetric:
            zero = torch.full_like(scale, (self.maxq + 1) / 2)
        else:
            zero = torch.round(-xmin / scale)

        if self.per_token:
            scale = scale.view(x_shape[:-1] + (1,))
            zero = zero.view(x_shape[:-1] + (1,))
        else:
            scale = scale.flatten()
            zero = zero.flatten()

        return scale, zero

    def forward(self, x: torch.Tensor):
        scale, zero = self.get_qparams(x)
        x = QuantizeDequantizeSTEFunction.apply(x, scale, zero, self.maxq)
        return self.linear(x)
