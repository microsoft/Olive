# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math
from abc import abstractmethod

import torch
import torch.nn as nn

from olive.common.quant.utils import WeightQuantizer


class QuantModule(nn.Module):
    """A base class for quantized modules.

    Only supports 2D weight tensors for now. Quantization axis is assumed to be the last dimension.
    - 4-bit and 8-bit quantization
    - Symmetric and asymmetric quantization
    - Per-channel and groupwise quantization

    For blockwise quantization, it has the following restrictions:
    - The size of the last dimension must be divisible by the group size. This is to avoid needing to pad the weights.
      Padding is easy but it complicates compatibility between contrib ops and QDQ representations, and weight tieing.
    - group size must be >= 16 and power of 2. For compatibility with the contrib ops.
    # TODO(jambayk): extend to support more general cases if needed.
    """

    def __init__(
        self,
        rows: int,
        cols: int,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize QuantLinear layer.

        Args:
            rows: Number of rows in the weight matrix
            cols: Number of columns in the weight matrix
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, >0: groupwise)
            device: Device to place tensors on
            dtype: Data type for scales and bias

        """
        super().__init__()
        if bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")
        if group_size != -1 and (group_size < 16 or (group_size & (group_size - 1)) != 0):
            raise ValueError("For blockwise quantization, group_size must be >= 16 and power of 2")

        self.rows = rows
        self.cols = cols
        self.quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        self.device = device
        self.dtype = dtype
        self.packing_factor = 8 // bits

        # using the same layout and packing as auto-gptq
        # TODO(jambayk): consider other packing schemes
        self.register_buffer(
            "qweight",
            torch.zeros(
                # rows X cols, packed as uint8 along last dim
                (self.rows, math.ceil(self.cols / self.packing_factor)),
                dtype=torch.uint8,
                device=device,
            ),
        )
        scale_shape = self.quantizer.get_qparam_shape((self.rows, self.cols))
        self.register_buffer(
            "scales",
            torch.zeros(scale_shape, dtype=dtype, device=device),
        )
        if symmetric:
            self.qzeros = None
        else:
            self.register_buffer(
                "qzeros",
                torch.zeros(
                    (scale_shape[0], math.ceil(scale_shape[1] / self.packing_factor)),
                    dtype=torch.uint8,
                    device=device,
                ),
            )

    @classmethod
    def from_tensors(
        cls,
        weight: torch.Tensor,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        scales: torch.device | None = None,
        zero_points: torch.device | None = None,
        quantized: bool = False,
        **kwargs,
    ) -> QuantModule:
        """Create a QuantLinear layer from an existing nn.Linear layer.

        Args:
            weight: The weight tensor. Expected to be 2D (unsigned, in range [0, 2^bits - 1] if quantized is True).
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, >0: groupwise)
            scales: Optional precomputed scales for quantization
            zero_points: Optional precomputed zero points for quantization (unsigned, in range [0, 2^bits - 1]).
            quantized: Whether the provided weight is already quantized
            kwargs: Additional keyword arguments to pass to the QuantModule constructor

        Returns:
            A QuantModule instance with quantized weight

        """
        if quantized:
            if scales is None or zero_points is None:
                raise ValueError("scales and/or zero_points missing for quantized weight")
            qweight = weight
        else:
            quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)

            # compute quantization parameters if not provided
            if scales is None:
                scales, zero_points = quantizer.find_qparams(weight)
            else:
                scales = scales.to(weight.device).to(weight.dtype)
                zero_points = zero_points.to(weight.device).to(torch.int32)

            # quantize weights
            qweight = quantizer.quantize(weight, scales, zero_points)

        rows, cols = qweight.shape
        qmodule = cls(
            rows,
            cols,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            device=qweight.device,
            dtype=scales.dtype,
            **kwargs,
        )
        qmodule.qweight = qmodule._pack(qweight.to(torch.int32)).contiguous()
        scale_shape = qmodule.quantizer.get_qparam_shape((rows, cols))
        qmodule.scales = scales.reshape(scale_shape)
        if not symmetric:
            qmodule.qzeros = qmodule._pack(zero_points.to(torch.int32).reshape(scale_shape)).contiguous()
        elif zero_points is not None and not torch.all(zero_points == qmodule.quantizer.midq):
            raise ValueError("Zero points must be None or equal to midq for symmetric quantization")
        return qmodule

    @torch.no_grad()
    def unpack_and_dequantize(
        self, qweight: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None
    ) -> torch.Tensor:
        """Unpack and dequantize the given quantized weight tensor.

        Returns:
            The dequantized weight tensor.

        """
        qweight = self._unpack(qweight, (qweight.shape[0], self.cols))
        if zero_points is not None:
            zero_points = self._unpack(zero_points, scales.shape)
        else:
            zero_points = torch.full_like(
                scales,
                self.quantizer.midq,
                dtype=torch.int32,
            )
        return self.quantizer.dequantize(qweight, scales, zero_points)

    @torch.no_grad()
    def _pack(self, tensor: torch.Tensor) -> torch.Tensor:
        """Pack a tensor of quantized weights or zero points into uint8 values.

        Values are expected to be unsigned in the range [0, 2^bits - 1] of dtype int32.

        Args:
            tensor: The tensor to pack. Will be packed along the last dimension
            bits: Number of bits used for quantization (4 or 8)

        Returns:
            The packed tensor.

        """
        assert tensor.dtype == torch.int32, "Input tensor must be of dtype int32"
        assert tensor.min() >= self.quantizer.minq, "Input tensor values must be non-negative for unsigned quantization"
        assert tensor.max() <= self.quantizer.maxq, "Input tensor values must not exceed max quantization value"

        # padd if necessary to ensure tensor shape is divisible by packing_factor
        num_padding = (-tensor.shape[-1]) % self.packing_factor
        if num_padding > 0:
            pad = (0, num_padding, 0, 0)
            tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
        packed_size = tensor.shape[-1] // self.packing_factor
        tensor = tensor.to(torch.uint8)

        packed_tensor = torch.zeros(
            (tensor.shape[0], packed_size),
            dtype=torch.uint8,
            device=tensor.device,
        )
        for i in range(self.packing_factor):
            packed_tensor |= tensor[:, i :: self.packing_factor] << self.quantizer.bits * i

        return packed_tensor

    @torch.no_grad()
    def _unpack(self, packed_tensor: torch.Tensor, shape: tuple[int, int]) -> torch.Tensor:
        """Unpack a tensor of packed uint8 values.

        Args:
            packed_tensor: The packed tensor to unpack. Will be unpacked along the last dimension.
            shape: The original shape of the tensor before packing

        Returns:
            The unpacked tensor of dtype int32.

        """
        assert packed_tensor.dtype == torch.uint8, "Input tensor must be of dtype uint8"

        wf = torch.arange(0, 8, self.quantizer.bits, device=packed_tensor.device, dtype=torch.uint8).unsqueeze(0)

        unpacked_tensor = torch.bitwise_right_shift(packed_tensor.unsqueeze(2), wf.unsqueeze(0))
        unpacked_tensor = unpacked_tensor.reshape(packed_tensor.shape[0], -1)
        unpacked_tensor = unpacked_tensor[:, : shape[1]]
        return torch.bitwise_and(unpacked_tensor, self.quantizer.maxq).to(torch.int32)

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the quantized module.

        Args:
            x: The input tensor.

        Returns:
            The output tensor.

        """


class QuantLinear(QuantModule):
    """Quantized Linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        bias: bool = True,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize QuantLinear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, >0: groupwise)
            bias: Whether to include bias
            device: Device to place tensors on
            dtype: Data type for scales and bias

        """
        super().__init__(
            rows=out_features,
            cols=in_features,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            device=device,
            dtype=dtype,
        )

        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=dtype, device=device),
            )
        else:
            self.bias = None

    @classmethod
    def from_module(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        scales: torch.device | None = None,
        zero_points: torch.device | None = None,
    ) -> QuantLinear:
        """Create a QuantLinear layer from an existing nn.Linear layer.

        Args:
            linear: The nn.Linear layer to convert
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, 0: per-tensor, >0: groupwise)
            scales: Optional precomputed scales for quantization
            zero_points: Optional precomputed zero points for quantization (unsigned, in range [0, 2^bits - 1]).

        Returns:
            A QuantLinear instance with quantized weights and scales

        """
        qlinear = cls.from_tensors(
            weight=linear.weight,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            scales=scales,
            zero_points=zero_points,
            bias=linear.bias is not None,
        )
        if linear.bias is not None:
            qlinear.bias = linear.bias.clone()
        return qlinear

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the quantized linear layer.

        Args:
            x: Input tensor of shape (..., in_features)

        Returns:
            Output tensor of shape (..., out_features)

        """
        assert x.shape[-1] == self.cols, f"Input shape {x.shape} does not match in_features {self.cols}"
        x_dtype = x.dtype

        # unpack weights and zero points
        weight = self.unpack_and_dequantize(self.qweight, self.scales, self.qzeros)
        return nn.functional.linear(x, weight, bias=self.bias).to(x_dtype)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.cols}, out_features={self.rows}, bits={self.quantizer.bits},"
            f" symmetric={self.quantizer.symmetric}, group_size={self.quantizer.group_size},"
            f" bias={self.bias is not None}"
        )


class QuantEmbedding(QuantModule):
    """Quantized Embedding layer."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        device: torch.device | None = None,
        dtype: torch.dtype = torch.float32,
        padding_idx: int | None = None,
    ):
        """Initialize QuantEmbedding layer.

        Args:
            num_embeddings: Number of embeddings
            embedding_dim: Dimension of each embedding
            padding_idx: Index of the padding token
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, >0: groupwise)
            device: Device to place tensors on
            dtype: Data type for scales

        """
        super().__init__(
            rows=num_embeddings,
            cols=embedding_dim,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            device=device,
            dtype=dtype,
        )
        self.padding_idx = padding_idx

    @classmethod
    def from_module(
        cls,
        embedding: nn.Embedding,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        scales: torch.device | None = None,
        zero_points: torch.device | None = None,
    ) -> QuantEmbedding:
        """Create a QuantEmbedding layer from an existing nn.Embedding layer.

        Args:
            embedding: The nn.Embedding layer to convert
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, >0: groupwise)
            scales: Optional precomputed scales for quantization
            zero_points: Optional precomputed zero points for quantization (unsigned, in range [0, 2^bits - 1]).

        Returns:
            A QuantEmbedding instance with quantized weights and scales

        """
        return cls.from_tensors(
            weight=embedding.weight,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            scales=scales,
            zero_points=zero_points,
            padding_idx=embedding.padding_idx,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the quantized embedding layer.

        Args:
            x: Input tensor of shape (...,)

        """
        qweight = nn.functional.embedding(x, self.qweight, padding_idx=self.padding_idx)
        scales = nn.functional.embedding(x, self.scales, padding_idx=self.padding_idx)
        if not self.quantizer.symmetric:
            qzeros = nn.functional.embedding(x, self.qzeros, padding_idx=self.padding_idx)
        else:
            qzeros = None
        return self.unpack_and_dequantize(qweight, scales, qzeros)

    def extra_repr(self) -> str:
        return (
            f"{self.rows}, {self.cols}, bits={self.quantizer.bits},"
            f" symmetric={self.quantizer.symmetric}, group_size={self.quantizer.group_size},"
            f" padding_idx={self.padding_idx}"
        )
