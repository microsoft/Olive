# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from typing import Optional

import torch
import torch.nn as nn

from olive.common.quant.utils import WeightQuantizer


class QuantLinear(nn.Module):
    """Quantized Linear layer.

    Supports:
    - 4-bit and 8-bit quantization
    - Symmetric and asymmetric quantization
    - Per-tensor, per-channel, and groupwise quantization
    - Optional bias
    - Packed storage for 4-bit weights and zero points
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        bias: bool = True,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        """Initialize QuantLinear layer.

        Args:
            in_features: Size of input features
            out_features: Size of output features
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, 0: per-tensor, >0: groupwise)
            bias: Whether to include bias
            device: Device to place tensors on
            dtype: Data type for scales and bias

        """
        super().__init__()
        if bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {bits}")

        self.in_features = in_features
        self.out_features = out_features
        self.quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        self.device = device
        self.dtype = dtype
        self.packing_factor = 32 // bits

        # using the same layout and packing as auto-gptq
        # TODO(jambayk): consider other packing schemes
        self.register_buffer(
            "qweight",
            torch.zeros(
                # in_features X out_features, packed as int32 along dim 0
                (math.ceil(self.in_features / self.packing_factor), self.out_features),
                dtype=torch.int32,
                device=device,
            ),
        )
        scale_shape = self.quantizer.get_qparam_shape((out_features, in_features), transpose_out=True)
        self.register_buffer(
            "scales",
            torch.zeros(scale_shape, dtype=dtype, device=device),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(
                (scale_shape[0], math.ceil(scale_shape[1] / self.packing_factor)),
                dtype=torch.int32,
                device=device,
            ),
        )
        if bias:
            self.register_buffer(
                "bias",
                torch.zeros(out_features, dtype=dtype, device=device),
            )
        else:
            self.bias = None

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        scales: Optional[torch.Tensor] = None,
        zero_points: Optional[torch.Tensor] = None,
    ) -> "QuantLinear":
        """Create a QuantLinear layer from an existing nn.Linear layer.

        Args:
            linear: The nn.Linear layer to convert
            bits: Number of bits for quantization (4 or 8)
            symmetric: Whether to use symmetric quantization
            group_size: Quantization group size (-1: per-channel, 0: per-tensor, >0: groupwise)
            scales: Optional precomputed scales for quantization
            zero_points: Optional precomputed zero points for quantization. Must be unsigned and in the range [1, 2^bits - 1].

        Returns:
            A QuantLinear instance with quantized weights and scales

        """
        # pylint: disable=W0201
        qlinear = cls(
            in_features=linear.in_features,
            out_features=linear.out_features,
            bits=bits,
            symmetric=symmetric,
            group_size=group_size,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype,
        )

        # compute quantization parameters if not provided
        if scales is None:
            scales, zero_points = qlinear.quantizer.find_qparams(linear.weight)
        else:
            scales = scales.to(qlinear.device).to(linear.weight.dtype)
            zero_points = zero_points.to(qlinear.device).to(torch.int32)

        # quantize weights
        qweight = qlinear.quantizer.quantize(linear.weight, scales, zero_points)

        # pack and assign parameters
        qlinear.qweight = qlinear._pack_to_int32(qweight.t(), axis=0).contiguous()
        qlinear.scales = scales.clone().t().contiguous()
        zero_points -= 1
        qlinear.qzeros = qlinear._pack_to_int32(zero_points.t(), axis=1).contiguous()
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
        assert x.shape[-1] == self.in_features, f"Input shape {x.shape} does not match in_features {self.in_features}"
        x_dtype = x.dtype

        # unpack weights and zero points
        qweight = self._unpack_from_int32(self.qweight, (self.in_features, self.out_features), axis=0)
        qzeros = self._unpack_from_int32(self.qzeros, self.scales.shape, axis=1) + 1
        scales = self.scales

        if torch.onnx.is_in_onnx_export():
            out = QuantLinearExportFunction.apply(
                x,
                *self._pack_mnb_format(
                    qweight.detach(),
                    scales.detach(),
                    qzeros.detach(),
                ),
                self.quantizer.bits,
                self.quantizer.group_size,
                self.in_features,
                self.out_features,
            )
        else:
            if self.quantizer.group_size > 0:
                scales = self.scales.repeat_interleave(self.quantizer.group_size, dim=0)
                qzeros = qzeros.repeat_interleave(self.quantizer.group_size, dim=0)
            qweight = (qweight - qzeros) * scales
            out = torch.matmul(x, qweight)

        if self.bias is not None:
            out += self.bias

        return out.to(x_dtype)

    @torch.no_grad()
    def get_unpacked_qparams(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get unpacked quantization parameters.

        Returns:
            Tuple of (unpacked weights, scales, zero points)

        """
        qweight = self._unpack_from_int32(self.qweight, (self.in_features, self.out_features), axis=0)
        qzeros = self._unpack_from_int32(self.qzeros, self.scales.shape, axis=1) + 1
        scales = self.scales
        return qweight, scales, qzeros

    @torch.no_grad()
    def _pack_to_int32(self, tensor: torch.Tensor, axis: int = 0) -> torch.Tensor:
        """Pack a tensor of quantized weights or zero points into int32.

        Values are expected to be unsigned in the range [0, 2^bits - 1] of dtype int32.

        Args:
            tensor: The tensor to pack, expected to be of shape (num_groups, num_features)
            bits: Number of bits used for quantization (4 or 8)
            axis: Axis along which to pack the values into int32

        """
        assert tensor.dtype == torch.int32, "Input tensor must be of dtype int32"
        assert tensor.min() >= self.quantizer.minq, "Input tensor values must be non-negative for unsigned quantization"
        assert tensor.max() <= self.quantizer.maxq, "Input tensor values must not exceed max quantization value"

        # padd if necessary to ensure tensor shape is divisible by packing_factor
        num_padding = (-tensor.shape[axis]) % self.packing_factor
        if num_padding > 0:
            pad = (0, num_padding, 0, 0) if axis == 1 else (0, 0, 0, num_padding)
            tensor = torch.nn.functional.pad(tensor, pad, mode="constant", value=0)
        packed_size = tensor.shape[axis] // self.packing_factor

        # TODO(jambayk): consider using bitwise operations
        if axis == 0:
            packed_tensor = torch.zeros(
                (packed_size, tensor.shape[1]),
                dtype=torch.int32,
                device=tensor.device,
            )
            for i in range(self.packing_factor):
                packed_tensor |= tensor[i :: self.packing_factor, :] << self.quantizer.bits * i
        else:
            packed_tensor = torch.zeros(
                (tensor.shape[0], packed_size),
                dtype=torch.int32,
                device=tensor.device,
            )
            for i in range(self.packing_factor):
                packed_tensor |= tensor[:, i :: self.packing_factor] << self.quantizer.bits * i

        return packed_tensor

    @torch.no_grad()
    def _unpack_from_int32(self, packed_tensor: torch.Tensor, shape: tuple[int, int], axis: int = 0) -> torch.Tensor:
        """Unpack a tensor of packed int32 values back to quantized weights or zero points.

        Args:
            packed_tensor: The packed tensor to unpack, expected to be of shape (num_groups, num_features)
            shape: The original shape of the tensor before packing
            axis: Axis along which the values were packed

        """
        assert packed_tensor.dtype == torch.int32, "Input tensor must be of dtype int32"

        wf = torch.arange(0, 32, self.quantizer.bits, device=packed_tensor.device, dtype=torch.int32).unsqueeze(0)

        if axis == 0:
            unpacked_tensor = torch.bitwise_right_shift(packed_tensor.unsqueeze(1), wf.unsqueeze(-1))
            unpacked_tensor = unpacked_tensor.reshape(-1, packed_tensor.shape[1])
            unpacked_tensor = unpacked_tensor[: shape[0]]
        else:
            unpacked_tensor = torch.bitwise_right_shift(packed_tensor.unsqueeze(2), wf.unsqueeze(0))
            unpacked_tensor = unpacked_tensor.reshape(packed_tensor.shape[0], -1)
            unpacked_tensor = unpacked_tensor[:, : shape[1]]
        return torch.bitwise_and(unpacked_tensor, self.quantizer.maxq)

    @torch.no_grad()
    def _pack_mnb_format(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        print("here")
        """Pack the quantized weight, scales, and zero points into the format expected by MatMulNBits."""
        # convert to shape and dtype expected by MatMulNBits
        # QuantLinear has K X N layout, while MatMulNBits expects N X K
        assert self.quantizer.bits in [4, 8], "Only 4-bit and 8-bit quantization supported"
        qweight = qweight.t().to(torch.uint8)
        # expect the scales to already be in the expected float type
        scales = scales.t()
        qzeros = qzeros.t().to(torch.uint8)

        # shapes for packing
        bits = self.quantizer.bits
        n, k = qweight.shape
        k_pack = 8 // bits
        block_size = self.quantizer.group_size
        blob_size = (block_size + k_pack - 1) // k_pack
        k_blocks = (k + block_size - 1) // block_size

        # pad qweight to make the k dimension divisible by block_size
        padded_k = k_blocks * block_size
        pad_len = padded_k - k
        if pad_len > 0:
            qweight = torch.nn.functional.pad(qweight, (0, pad_len), value=0)
        # pack qweight
        if bits == 4:
            qweight = (qweight[:, 0::2] & 0xF) | ((qweight[:, 1::2] & 0xF) << 4)
        qweight = qweight.reshape(n, k_blocks, blob_size).contiguous()

        # pad qzeros to make the k dimension even
        qzeros = torch.nn.functional.pad(qzeros, (0, qzeros.shape[1] & 1), value=0)
        # pack qzeros
        if bits == 4:
            qzeros = (qzeros[:, 0::2] & 0xF) | ((qzeros[:, 1::2] & 0xF) << 4)
        qzeros = qzeros.flatten().contiguous()

        # flatten scales
        scales = scales.flatten().contiguous()
        print("here")

        return qweight, scales, qzeros

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, bits={self.quantizer.bits},"
            f" symmetric={self.quantizer.symmetric}, group_size={self.quantizer.group_size},"
            f" bias={self.bias is not None}"
        )


class QuantLinearExportFunction(torch.autograd.Function):
    """Function for QuantLinear layer to support export to MatMulNBits."""

    # pylint: disable=W0221
    @staticmethod
    def symbolic(g, x, qweight, scales, qzeros, bits, group_size, in_features, out_features):
        tensor_args = [x, qweight, scales, qzeros]
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
            output_type = x.type().with_sizes(input_shape[:-1] + [qweight.type().varyingSizes()[0]])
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
        if hasattr(torch.onnx, "ops"):
            # torch.onnx.ops was introduced in 2.8
            tensor_args = [x, qweight, scales, qzeros]
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
        else:
            return torch.zeros(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device)
