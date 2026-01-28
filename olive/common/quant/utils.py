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
            bits: Number of bits for quantization (2, 4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, 0: per-tensor, >0: groupwise)
            signed: Whether to use signed quantization (default is False, meaning unsigned)

        """
        assert bits in [2, 4, 8], "Only 4-bit and 8-bit quantization supported"
        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
        self.signed = signed

        self.maxq, self.minq = get_maxq_minq(self.bits, self.signed)
        self.midq = (self.maxq + self.minq + 1) // 2

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

    def get_qparam_shape(self, shape: tuple[int, int]) -> tuple[int, ...]:
        """Get the shapes for quantization parameters based on the input shape and group_size.

        Args:
            shape: The shape (out_features, in_features) of the tensor to quantize

        Returns:
            A tuple of shapes for scales and zero points

        """
        if self.group_size == 0:
            return (1, 1)
        return (shape[0], self.get_num_groups(shape))

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
            zero_points = torch.full_like(scales, self.midq)
        else:
            zero_points = torch.round(self.minq - min_val / scales)
        zero_points = zero_points.clamp(self.minq, self.maxq).to(torch.int32)

        return scales, zero_points

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

    def fake_quantize(
        self, tensor: torch.Tensor, scales: torch.Tensor | None = None, zero_points: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Fake quantize the given tensor using the provided scales and zero points.

        Args:
            tensor: The tensor to quantize. Expected to be 2D with shape (out_features, in_features).
            scales: The scales for quantization. If None, scales will be computed from the tensor.
            zero_points: The zero points for quantization. If None, zero points will be computed from the tensor.

        Returns:
            The fake quantized tensor.

        """
        if scales is None or zero_points is None:
            scales, zero_points = self.find_qparams(tensor)
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


def get_maxq_minq(bits: int, signed: bool) -> tuple[int, int]:
    """Get the maximum and minimum quantization values based on bits and signedness.

    Args:
        bits: Number of bits (4 or 8).
        signed: Whether the quantization is signed or unsigned.

    Returns:
        A tuple of (maxq, minq).

    """
    if signed:
        half = 1 << (bits - 1)
        minq = -half
        maxq = half - 1
    else:
        minq = 0
        maxq = (1 << bits) - 1
    return maxq, minq


@torch.no_grad()
def pack_to_uint8(tensor: torch.Tensor, bits: int) -> torch.Tensor:
    """Pack 2/4/8 bit tensor into uint8 tensor on the last dimension.

    Args:
        tensor: The 2D input tensor. Values are expected to be unsigned in the range [0, 2^bits - 1]
        bits: Number of bits (2, 4 or 8)

    Returns:
        A tensor of uint8 values with packed data

    """
    assert bits in [2, 4, 8], "Only 2-bit, 4-bit and 8-bit quantization supported"

    maxq, minq = get_maxq_minq(bits, signed=False)
    assert tensor.min() >= minq, "Input tensor values must not be less than min quantization value"
    assert tensor.max() <= maxq, "Input tensor values must not exceed max quantization value"

    packing_factor = 8 // bits

    # padd if necessary to ensure tensor shape is divisible by packing_factor
    num_padding = (-tensor.shape[-1]) % packing_factor
    if num_padding > 0:
        pad = (0, num_padding, 0, 0)
        tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
    packed_size = tensor.shape[-1] // packing_factor
    tensor = tensor.to(torch.uint8)

    packed_tensor = torch.zeros(
        (tensor.shape[0], packed_size),
        dtype=torch.uint8,
        device=tensor.device,
    )
    for i in range(packing_factor):
        packed_tensor |= tensor[:, i::packing_factor] << bits * i
    return packed_tensor


@torch.no_grad()
def unpack_from_uint8(packed_tensor: torch.Tensor, bits: int, shape: tuple[int, int]) -> torch.Tensor:
    """Unpack uint8 tensor into 2/4/8 bit tensor on the last dimension.

    Args:
        packed_tensor: The 2D input tensor with packed uint8 values
        bits: Number of bits (2, 4 or 8)
        shape: The original shape of the tensor before packing

    Returns:
        A tensor of int32 values with unpacked data

    """
    assert packed_tensor.dtype == torch.uint8, "Input tensor must be of dtype uint8"

    maxq, _ = get_maxq_minq(bits, signed=False)

    wf = torch.arange(0, 8, bits, device=packed_tensor.device, dtype=torch.uint8).unsqueeze(0)

    unpacked_tensor = torch.bitwise_right_shift(packed_tensor.unsqueeze(2), wf.unsqueeze(0))
    unpacked_tensor = unpacked_tensor.reshape(shape[0], -1)
    unpacked_tensor = unpacked_tensor[:, : shape[1]]
    return torch.bitwise_and(unpacked_tensor, maxq).to(torch.int32)
