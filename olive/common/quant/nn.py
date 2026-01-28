# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math
from abc import abstractmethod

import torch
import torch.nn as nn

from olive.common.quant.utils import WeightQuantizer, pack_to_uint8, unpack_from_uint8


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
        if bits not in [2, 4, 8]:
            raise ValueError(f"Only 2-bit, 4-bit and 8-bit quantization supported, got {bits}")
        if group_size != -1 and (group_size < 16 or (group_size & (group_size - 1)) != 0):
            raise ValueError("For blockwise quantization, group_size must be >= 16 and power of 2")
        if group_size != -1 and cols % group_size != 0:
            raise ValueError(
                f"For blockwise quantization, cols ({cols}) must be divisible by group_size ({group_size})"
            )

        self.rows = rows
        self.cols = cols
        self.quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        self.device = device
        self.dtype = dtype

        packing_factor = 8 // bits

        # using the same layout and packing as auto-gptq
        # TODO(jambayk): consider other packing schemes
        self.register_buffer(
            "qweight",
            torch.zeros(
                # rows X cols, packed as uint8 along last dim
                (self.rows, math.ceil(self.cols / packing_factor)),
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
                    (scale_shape[0], math.ceil(scale_shape[1] / packing_factor)),
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
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
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
        # pylint: disable=W0201
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

        qmodule = cls(
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            device=qweight.device,
            dtype=scales.dtype,
            **kwargs,
        )
        qmodule.qweight = pack_to_uint8(qweight, bits).contiguous()
        scale_shape = qmodule.quantizer.get_qparam_shape(qweight.shape)
        qmodule.scales = scales.reshape(scale_shape).contiguous()

        # enforce symmetric quantization constraints
        if symmetric and not torch.all(zero_points == qmodule.quantizer.midq):
            raise ValueError("Zero points must be equal to midq for symmetric quantization")

        if not symmetric:
            qmodule.qzeros = pack_to_uint8(zero_points.reshape(scale_shape), bits).contiguous()

        return qmodule

    @torch.no_grad()
    def unpack_and_dequantize(
        self, qweight: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor | None
    ) -> torch.Tensor:
        """Unpack and dequantize the given quantized weight tensor.

        Returns:
            The dequantized weight tensor.

        """
        qweight = unpack_from_uint8(qweight, self.quantizer.bits, (qweight.shape[0], self.cols))
        if zero_points is not None:
            zero_points = unpack_from_uint8(zero_points, self.quantizer.bits, scales.shape)
        else:
            zero_points = torch.full_like(
                scales,
                self.quantizer.midq,
                dtype=torch.int32,
            )
        return self.quantizer.dequantize(qweight, scales, zero_points)

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
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
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
            in_features=linear.in_features,
            out_features=linear.out_features,
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

        if torch.onnx.is_in_onnx_export():
            out = QuantLinearFunction.apply(
                x,
                self.qweight.reshape([*self.scales.shape, self.qweight.shape[-1] // self.scales.shape[-1]]),
                self.scales,
                self.qzeros,
                self.quantizer.bits,
                self.quantizer.group_size if self.quantizer.group_size > 0 else self.cols,
                self.cols,
                self.rows,
            )
            return out + self.bias if self.bias is not None else out

        x_dtype = x.dtype

        # unpack weights and zero points
        weight = self.unpack_and_dequantize(self.qweight, self.scales, self.qzeros)
        return nn.functional.linear(x, weight, bias=self.bias).to(x_dtype)  # pylint: disable=not-callable

    def extra_repr(self) -> str:
        return (
            f"in_features={self.cols}, out_features={self.rows}, bits={self.quantizer.bits},"
            f" symmetric={self.quantizer.symmetric}, group_size={self.quantizer.group_size},"
            f" bias={self.bias is not None}"
        )


# this is required for torchscript onnx export
class QuantLinearFunction(torch.autograd.Function):
    # pylint: disable=W0223,W0221
    @staticmethod
    def symbolic(g, x, qweight, scales, qzeros, bits, group_size, in_features, out_features):
        tensor_args = [x, qweight, scales]
        if qzeros is not None:
            tensor_args.append(qzeros)
        attrs = {
            "K_i": in_features,
            "N_i": out_features,
            "bits_i": bits,
            "block_size_i": group_size,
            "accuracy_level_i": 4,
        }

        output = g.op(
            "com.microsoft::MatMulNBits",
            *tensor_args,
            outputs=1,
            **attrs,
        )
        input_shape = x.type().varyingSizes()
        if input_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = x.type().with_sizes(input_shape[:-1] + [out_features])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bits: int,
        group_size: int,
        in_features: int,
        out_features: int,
    ):
        # there is no clean way to differentiate between torchscript and dynamo export
        # using FakeTensor as a proxy for dynamo export
        if hasattr(torch.onnx, "ops") and isinstance(x, torch._subclasses.FakeTensor):  # pylint: disable=W0212
            tensor_args = [x, qweight, scales]
            if qzeros is not None:
                tensor_args.append(qzeros)
            attrs = {
                "K": in_features,
                "N": out_features,
                "bits": bits,
                "block_size": group_size,
                "accuracy_level": 4,
            }
            return torch.onnx.ops.symbolic(
                "com.microsoft::MatMulNBits",
                tensor_args,
                attrs=attrs,
                dtype=x.dtype,
                shape=[*x.shape[:-1], out_features],
                version=1,
            )
        return torch.zeros([*x.shape[:-1], out_features], dtype=x.dtype, device=x.device)


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
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
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
            num_embeddings=embedding.num_embeddings,
            embedding_dim=embedding.embedding_dim,
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
        if torch.onnx.is_in_onnx_export():
            return QuantEmbeddingFunction.apply(
                x,
                self.qweight,
                self.scales,
                self.qzeros,
                self.quantizer.bits,
                self.quantizer.group_size if self.quantizer.group_size > 0 else self.cols,
                self.cols,
            )

        # reshape input to 1D so that look up results are 2D
        x_shape = x.shape
        x = x.reshape(-1)

        # look up quantized weights and qparams
        qweight = nn.functional.embedding(x, self.qweight, padding_idx=self.padding_idx)
        scales = nn.functional.embedding(x, self.scales, padding_idx=self.padding_idx)
        if self.qzeros is not None:
            qzeros = nn.functional.embedding(x, self.qzeros, padding_idx=self.padding_idx)
        else:
            qzeros = None

        # unpack and dequantize
        return self.unpack_and_dequantize(qweight, scales, qzeros).reshape(*x_shape, -1)

    def extra_repr(self) -> str:
        return (
            f"{self.rows}, {self.cols}, bits={self.quantizer.bits},"
            f" symmetric={self.quantizer.symmetric}, group_size={self.quantizer.group_size},"
            f" padding_idx={self.padding_idx}"
        )


# this is required for torchscript onnx export
class QuantEmbeddingFunction(torch.autograd.Function):
    # pylint: disable=W0223,W0221
    @staticmethod
    def symbolic(g, x, qweight, scales, qzeros, bits, group_size, embedding_dim):
        tensor_args = [qweight, x, scales]
        if qzeros is not None:
            tensor_args.append(qzeros)
        attrs = {"bits_i": bits, "block_size_i": group_size}

        output = g.op(
            "com.microsoft::GatherBlockQuantized",
            *tensor_args,
            outputs=1,
            **attrs,
        )
        input_shape = x.type().varyingSizes()
        if input_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = scales.type().with_sizes([*input_shape, embedding_dim])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
        bits: int,
        group_size: int,
        embedding_dim: int,
    ):
        if hasattr(torch.onnx, "ops") and isinstance(x, torch._subclasses.FakeTensor):  # pylint: disable=W0212
            tensor_args = [qweight, x, scales]
            if qzeros is not None:
                tensor_args.append(qzeros)
            attrs = {"bits": bits, "block_size": group_size}
            return torch.onnx.ops.symbolic(
                "com.microsoft::GatherBlockQuantized",
                tensor_args,
                attrs=attrs,
                dtype=scales.dtype,
                shape=[*x.shape, embedding_dim],
                version=1,
            )
        return torch.zeros((*x.shape, embedding_dim), dtype=scales.dtype, device=x.device)
