# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import torch


# TODO(jambayk): consider supporting transposed weights, useful for onnx weights
class WeightQuantizer:
    """Class to quantize weight tensors."""

    def __init__(self, bits: int = 4, symmetric: bool = True, group_size: int = 0, signed: bool = False):
        """Initialize the quantizer with parameters.

        Args:
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, 0: per-tensor, >0: groupwise)
            signed: Whether to use signed quantization (default is False, meaning unsigned)

        """
        assert bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.signed = signed

        if self.signed:
            half = 1 << (self.bits - 1)
            self.minq = -half
            self.maxq = half - 1
        else:
            self.minq = 0
            self.maxq = (1 << self.bits) - 1

    def get_num_groups(self, shape: tuple[int, int]) -> int:
        """Get the number of groups for quantization based on the input shape and group_size.

        Args:
            shape: The shape (out_features, in_features) of the tensor to quantize

        Returns:
            The number of groups for quantization

        """
        if self.group_size == 0:
            raise ValueError("group_size must be greater than 0 for groupwise quantization")
        group_size = self.group_size if self.group_size > 0 else shape[1]
        assert shape[1] % group_size == 0, f"in_features {shape[1]} must be divisible by group_size {group_size}"
        return shape[1] // group_size

    def get_qparam_shape(self, shape: tuple[int, int], transpose_out: bool = False) -> tuple[int, ...]:
        """Get the shapes for quantization parameters based on the input shape and group_size.

        Args:
            shape: The shape (out_features, in_features) of the tensor to quantize
            transpose_out: Whether the output is transposed

        Returns:
            A tuple of shapes for scales and zero points

        """
        if self.group_size == 0:
            return (1, 1)
        return (shape[0], self.get_num_groups(shape)) if not transpose_out else (self.get_num_groups(shape), shape[0])

    @torch.no_grad()
    def find_qparams(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Find quantization parameters (scale and zero point) for the given tensor.

        Args:
            tensor: The tensor to quantize. Expected to be 2D with shape (out_features, in_features).

        Returns:
            A tuple of (scale, zero_point)

        """
        tensor, _ = self._reshape_tensor(tensor)

        # calculate min and max
        tmp = torch.zeros(tensor.shape[0:-1], device=tensor.device, dtype=tensor.dtype)
        min_val = torch.minimum(tensor.min(-1)[0], tmp)
        max_val = torch.maximum(tensor.max(-1)[0], tmp)

        if self.symmetric:
            max_val = torch.maximum(abs(min_val), max_val)
            tmp = min_val < 0
            if torch.any(tmp):
                min_val[tmp] = -max_val[tmp]

        tmp = (min_val == 0) & (max_val == 0)
        min_val[tmp] = -1
        max_val[tmp] = 1

        scales = (max_val - min_val) / (self.maxq - self.minq)
        if self.symmetric:
            zero_points = torch.full_like(scales, (self.maxq + self.minq + 1) / 2)
        else:
            zero_points = torch.round(self.minq - min_val / scales)
        zero_points = zero_points.clamp(self.minq, self.maxq).to(torch.int32)

        return scales, zero_points

    # TODO(jambayk): consider moving quantize and dequantize into generic helper functions
    @torch.no_grad()
    def quantize(self, tensor: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor) -> torch.Tensor:
        """Quantize the given tensor using the provided scales and zero points.

        Args:
            tensor: The tensor to quantize. Expected to be 2D with shape (out_features, in_features).
            scales: The scales for quantization.
            zero_points: The zero points for quantization.

        Returns:
            The quantized tensor.

        """
        tensor, shape = self._reshape_tensor(tensor)

        # apply quantization
        q_tensor = torch.round(tensor / scales.unsqueeze(-1) + zero_points.unsqueeze(-1))
        q_tensor = q_tensor.clamp(self.minq, self.maxq).to(torch.int32)

        return q_tensor.reshape(shape)

    @torch.no_grad()
    def dequantize(self, q_tensor: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor) -> torch.Tensor:
        """Dequantize the given quantized tensor using the provided scales and zero points.

        Args:
            q_tensor: The quantized tensor to dequantize. Expected to be 2D with shape (out_features, in_features).
            scales: The scales for dequantization.
            zero_points: The zero points for dequantization.

        Returns:
            The dequantized tensor.

        """
        q_tensor, shape = self._reshape_tensor(q_tensor)

        # apply dequantization
        # both q_tensor and zero_points should be int32, so no need to worry about overflow
        tensor = (q_tensor - zero_points.unsqueeze(-1)) * scales.unsqueeze(-1)
        return tensor.reshape(shape)

    def fake_quantize(self, tensor: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor) -> torch.Tensor:
        """Fake quantize the given tensor using the provided scales and zero points.

        Args:
            tensor: The tensor to quantize. Expected to be 2D with shape (out_features, in_features).
            scales: The scales for quantization.
            zero_points: The zero points for quantization.

        Returns:
            The fake quantized tensor.

        """
        q_tensor = self.quantize(tensor, scales, zero_points)
        return self.dequantize(q_tensor, scales, zero_points)

    def _reshape_tensor(self, tensor: torch.Tensor) -> tuple[torch.Tensor, tuple[int, ...]]:
        """Reshape the tensor based on the group size.

        Args:
            tensor: The tensor to reshape.

        Returns:
            The reshaped tensor and its original shape.

        """
        shape = tensor.shape
        if self.group_size == 0:
            tensor = tensor.reshape(1, 1, -1)
        else:
            tensor = tensor.reshape(shape[0], self.get_num_groups(shape), -1)
        return tensor, shape
