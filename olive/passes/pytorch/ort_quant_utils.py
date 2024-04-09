# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
import os

import torch
import torch.nn as nn


def _packed4bit_to_float(data, axis=-1):
    # unpack 4bit tensor to float
    shape = data.size()
    new_shape = shape[:-1] + (shape[-1] * 2,)
    data = data.repeat_interleave(2, axis=axis)
    data = data.reshape(-1, 2)
    data = torch.bitwise_right_shift(data, torch.tensor([0, 4], dtype=torch.uint8, device=data.device))
    data = torch.bitwise_and(data, 0x0F)
    return data.reshape(*new_shape)


def _pt_linear(x, qweight, qscales, qzeros, in_features, out_features, bits, group_size):
    # pytorch unfused implementation of quantized linear
    qweight = _packed4bit_to_float(qweight.reshape(out_features, -1))
    qscales = qscales.reshape(out_features, -1).repeat_interleave(group_size, dim=-1)
    qzeros = _packed4bit_to_float(
        qzeros.reshape(out_features, in_features // 8 * bits // group_size)
    ).repeat_interleave(group_size, dim=-1)
    qzeros = -qscales * qzeros
    qweight = qweight * qscales
    qweight = qweight + qzeros
    return torch.matmul(x, qweight.T)


def _pack_on_row_fast_anybit(pack_tensor, ori_int_tensor, bits):
    device = pack_tensor.device
    need_transpose = False
    if pack_tensor.shape[0] != ori_int_tensor.shape[0]:
        need_transpose = True
        ori_int_tensor = ori_int_tensor.T
    pack_tensor.mul_(0)
    wf = torch.arange(0, bits, device=device).view(1, 1, -1)
    out = torch.bitwise_right_shift(ori_int_tensor.unsqueeze(-1), wf)
    torch.bitwise_and(out, 1, out=out)
    out = out.reshape(ori_int_tensor.shape[0], -1, 32)
    wf1 = torch.arange(0, 32, 1, device=device).view(1, 1, -1)
    out = torch.bitwise_left_shift(out, wf1)
    out = out.sum(dim=-1).int()

    if need_transpose:
        out = out.T.contiguous()
    pack_tensor.copy_(out)


def _dequantize_blockwise_4bits(quant_values, scale, zero_point, rows, cols):
    expand_quant_value = (
        quant_values.unsqueeze(-1) >> torch.tensor([[[[0, 4]]]], dtype=torch.int32, device=quant_values.device)
    ) & 0x0F
    expand_quant_value = expand_quant_value.reshape(*quant_values.shape[:-1], -1)
    aligned_scale = scale.reshape(*quant_values.shape[:-1], 1)
    expand_zero_point = (
        zero_point.unsqueeze(-1) >> torch.tensor([[[[0, 4]]]], dtype=torch.int32, device=quant_values.device)
    ) & 0x0F
    expand_zero_point = expand_zero_point.reshape(*quant_values.shape[:-1], -1)
    float_values = ((expand_quant_value - expand_zero_point) * aligned_scale).to(scale.dtype)
    float_values = float_values.reshape(cols, -1)
    if rows != float_values.shape[-1]:
        float_values = float_values[:, :rows]
        expand_zero_point = expand_zero_point[:, :rows]
    if expand_zero_point.ndim == 3:
        expand_zero_point = expand_zero_point.squeeze(-1)
    if aligned_scale.ndim == 3:
        aligned_scale = aligned_scale.squeeze(-1)

    return float_values, expand_zero_point, aligned_scale


def _general_pack_on_row(pack_tensor, ori_int32_tensor, bits):
    # general pack for any bits
    assert pack_tensor.shape[0] == ori_int32_tensor.shape[0] or pack_tensor.shape[1] == ori_int32_tensor.shape[1]
    pack_tensor.mul_(0)
    if bits in [2, 4, 8]:
        _pack_on_row_fast_248bit(pack_tensor, ori_int32_tensor, bits)
    _pack_on_row_fast_anybit(pack_tensor, ori_int32_tensor, bits)


def _pack_on_row_fast_248bit(pack_tensor, ori_int_tensor, bits):
    if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
        ori_int_tensor = ori_int_tensor.T
        pack_tensor = pack_tensor.T
    compress_ratio = 32 // bits
    i = 0
    row = 0
    while row < pack_tensor.shape[0]:
        for j in range(i, i + compress_ratio):
            pack_tensor[row:] |= ori_int_tensor[j::compress_ratio] << (bits * (j - i))


class QuantLinearTorchFunction(torch.autograd.Function):
    # pylint: disable=W0223,W0221
    @staticmethod
    def symbolic(g, x, qself_qweight, qself_scales, qself_qzeros, bits, groupsize, in_features, out_features):
        output = g.op(
            "com.microsoft::MatMulNBits",
            x,
            qself_qweight,
            qself_scales,
            qself_qzeros,
            outputs=1,
            K_i=in_features,
            N_i=out_features,
            bits_i=bits,
            block_size_i=groupsize,
        )
        input_shape = x.type().varyingSizes()
        if input_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = x.type().with_sizes(input_shape[:-1] + [qself_qweight.type().varyingSizes()[0]])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(ctx, x, qself_qweight, qself_scales, qself_qzeros, bits, groupsize, in_features, out_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device)
        x_dtype = x.dtype
        x = x.to(qself_scales)
        return _pt_linear(x, qself_qweight, qself_scales, qself_qzeros, in_features, out_features, bits, groupsize).to(
            x_dtype
        )


def _quant_linear_forward(inputs, qweight, scales, qzeros, bits, groupsize, in_features, out_features):
    assert bits == 4, "Only 4 bits are supported."
    return QuantLinearTorchFunction().apply(inputs, qweight, scales, qzeros, bits, groupsize, in_features, out_features)


class QuantLinearORT(nn.Module):
    # pylint: disable=W0201
    def __init__(self, bits, groupsize, infeatures, outfeatures, bias, *args, **kwargs):
        super().__init__()
        if bits not in [2, 3, 4, 5, 6, 7, 8]:
            raise ValueError("Only 2,4,5,6,7,8 bits are supported.")
        self.infeatures = infeatures
        self.outfeatures = outfeatures
        self.bits = bits
        self.orig_fp_weight = None
        self.maxq = 2**self.bits - 1
        self.groupsize = groupsize if groupsize != -1 else infeatures

        self.register_buffer(
            "qweight",
            torch.zeros((outfeatures, infeatures // self.groupsize, self.groupsize // (8 // bits)), dtype=torch.uint8),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(infeatures // self.groupsize) * (outfeatures // 8 * self.bits)), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(infeatures / self.groupsize) * outfeatures), dtype=torch.float)
        )
        self.register_buffer("g_idx", torch.tensor([i // self.groupsize for i in range(infeatures)], dtype=torch.int32))
        if bias:
            self.register_buffer("bias", torch.zeros((outfeatures), dtype=torch.float))
        else:
            self.bias = None

    def forward(self, x):
        out = _quant_linear_forward(
            x, self.qweight, self.scales, self.qzeros, self.bits, self.groupsize, self.infeatures, self.outfeatures
        )
        return out + self.bias if self.bias is not None else out

    def pack(self, linear, scales, zeros, g_idx=None):
        # function to transform float weights to quantized integer weights and pack them
        # using one byte to save multiple subbyte data points
        layer_weight = linear.weight.data

        self.g_idx = g_idx.clone() if g_idx is not None else self.g_idx
        intweight = self._quant_weight(layer_weight, scales, zeros, g_idx, need_transpose=False)

        qzeros = zeros.t().contiguous()

        if self.bits in [2, 4, 8]:
            self._pack_for_even_bits(intweight, qzeros)
        else:
            self._pack_for_odd_bits(intweight, qzeros)

    def _unpack(self):
        float_values, zero_point, scale = _dequantize_blockwise_4bits(
            self.qweight, self.scales, self.qzeros, self.infeatures, self.outfeatures
        )
        float_values = float_values.contiguous()
        zero_point = zero_point.T.contiguous()
        scale = scale.T.contiguous()
        return float_values, zero_point, scale

    def _quant_weight(self, weight, scales, zeros, g_idx=None, need_transpose=True):
        # quantize float weight to integer weight
        scales = scales.t().contiguous()
        zeros = zeros.t().contiguous()
        g_idx = self.g_idx.long()
        scale_zeros = zeros * scales
        self.scales = scales.clone() if self.scales.sum() == 0 else self.scales

        scale_mat = scales[g_idx]
        scale_zeros_mat = scale_zeros[g_idx]
        intweight_t = torch.round((weight.T + scale_zeros_mat) / scale_mat).to(torch.int)

        if not need_transpose:
            return intweight_t
        return intweight_t.T

    def _pack_for_even_bits(self, intweight, intzeros_T):
        # pack multiple integer weights into one byte, this function is optimized for
        # even bits
        self.act_order = self.g_idx[: self.groupsize // self.bits].sum().item() != 0
        assert self.act_order is False, "act_order=True is not supported"
        intzeros_pt = intzeros_T.T.byte()
        scales_pt = self.scales.T
        intweight_pt = intweight.byte()
        block_size = self.groupsize

        rows, cols = intweight_pt.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            intweight_pt = torch.nn.functional.pad(intweight_pt, (0, 0, 0, pad_len), "constant", 0)
            intzeros_pt = torch.nn.functional.pad(intzeros_pt, (0, 0, 0, pad_len), "constant", 0)

        intzeros_pt = (intzeros_pt[:, 0::2]) | (intzeros_pt[:, 1::2] << 4)
        intzeros_pt = intzeros_pt.reshape(-1)

        intweight_pt_t = intweight.T
        intweight_pt_t = (intweight_pt_t[:, 0::2]) | (intweight_pt_t[:, 1::2] << 4)
        intweight_pt_t = intweight_pt_t.reshape(cols, k_blocks, blob_size)

        scales_pt = scales_pt.reshape(-1)

        assert self.qweight.shape == intweight_pt_t.shape
        assert self.qzeros.shape == intzeros_pt.shape

        self.scales = scales_pt.contiguous()
        self.qweight = intweight_pt_t.contiguous().byte()
        self.qzeros = intzeros_pt.contiguous().byte()

    def _pack_for_odd_bits(self, intweight, intzeros):
        device = intweight.device
        qweight = torch.zeros(
            ((intweight.shape[0] * self.bits + 31) // 32, intweight.shape[1]), dtype=torch.int32, device=device
        )

        _general_pack_on_row(qweight, intweight, self.bits)
        self.qweight = qweight

        self._pack_qzeros_odd(intzeros)

        if self.orig_fp_weight is not None:
            fw, _, _ = self._unpack()
            assert (fw == self.orig_fp_weight).all()

    def _pack_qzeros_odd(self, intzeros):
        # why -1?
        # zeros_cuda = (zeros - 1).to(device).int()
        compatible_with_autogptq = int(os.environ.get("COMPATIBLE_WITH_AUTOGPTQ", "0"))
        device = intzeros.device
        zeros = intzeros - compatible_with_autogptq
        max_num_in_bits = 2**self.bits - 1
        zeros = (zeros.byte() & max_num_in_bits).int()
        qzeros = torch.zeros(
            (intzeros.shape[0], (intzeros.shape[1] * self.bits + 31) // 32), dtype=torch.int32, device=device
        )

        qzeros = qzeros.T.contiguous()
        zeros = zeros.T.contiguous()
        self._general_pack_on_row(qzeros, zeros, self.bits)

        self.qzeros = qzeros.T.contiguous()
