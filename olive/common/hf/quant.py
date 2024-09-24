# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import math
from typing import Callable, Dict, Tuple

import torch
import transformers

from olive.common.utils import set_attr


def get_quantization_info(model: torch.nn.Module):
    """Get the quantization info from the model."""
    if (
        hasattr(model, "quantization_method")
        and hasattr(model, "config")
        and hasattr(model.config, "quantization_config")
    ):
        return model.quantization_method, model.config.quantization_config
    return None, None


def is_quantized_model(model: torch.nn.Module):
    """Check if the model is quantized."""
    return get_quantization_info(model)[0] is not None


def get_auto_awq_qlinear_cls(quantization_config):
    """Get the right AutoAWQQuantLinear class based on the quantization config."""
    if transformers.utils.import_utils.is_auto_awq_available():
        from awq.modules.linear import WQLinear_GEMM

        return WQLinear_GEMM
    return None


# copied from https://github.com/huggingface/peft/blob/f4cf170a9c51d822f950cde0a0e1c87dc013403a/src/peft/utils/other.py#L579
def get_auto_gptq_qlinear_cls(quantization_config):
    """Get the right AutoGPTQQuantLinear class based on the quantization config."""
    if transformers.utils.import_utils.is_auto_gptq_available():
        from auto_gptq.utils.import_utils import dynamically_import_QuantLinear

        desc_act = quantization_config.desc_act
        group_size = quantization_config.group_size
        bits = quantization_config.bits
        if hasattr(quantization_config, "use_exllama"):
            use_exllama = quantization_config.use_exllama
        else:
            use_exllama = not quantization_config.disable_exllama
        if hasattr(quantization_config, "exllama_config"):
            exllama_version = quantization_config.exllama_config["version"]
        else:
            exllama_version = 1
        return dynamically_import_QuantLinear(
            use_triton=False,
            desc_act=desc_act,
            group_size=group_size,
            bits=bits,
            disable_exllama=not (use_exllama and exllama_version == 1),
            disable_exllamav2=not (use_exllama and exllama_version == 2),
        )
    return None


def get_bnb_qlinear_cls(quantization_config):
    """Get the right BNBQuantLinear class based on the quantization config."""
    if transformers.utils.import_utils.is_bitsandbytes_available() and quantization_config.load_in_4bit:
        from bitsandbytes.nn import Linear4bit

        return Linear4bit
    return None


def _replace_qlinear_modules(
    model: torch.nn.Module, mapping: Dict[str, Tuple[Callable, Callable]]
) -> Tuple[torch.nn.Module, bool]:
    """Make the model export compatible by replacing the quantized linear layers with 4-bit versions.

    :param model: the model to make export compatible. Only modified if model is quantized using a supported method.
    :param mapping: the mapping from quantization method to
        - function to get qlinear class from quantization config
        - function to get new class to replace qlinear modules with
    :return model: the modified model
    :return modified: whether the model was modified
    """
    quantization_method, quantization_config = get_quantization_info(model)

    if quantization_method not in mapping:
        return model, False

    # get the QuantLinear class to look for in the model
    qlinear_class = mapping[quantization_method][0](quantization_config)
    if qlinear_class is None:
        return model, False

    # search and replace
    modified = False
    for name, module in model.named_modules():
        if isinstance(module, qlinear_class):
            new_module = mapping[quantization_method][1](module)
            set_attr(model, name, new_module)
            modified = True

    return model, modified


# Dequantization functions


@torch.no_grad()
def dequantize_auto_gptq_qlinear(qlinear) -> torch.nn.Linear:
    from auto_gptq.nn_modules.qlinear.qlinear_marlin import unpack_4bit_to_32bit_signed

    linear = torch.nn.Linear(
        qlinear.infeatures,
        qlinear.outfeatures,
        bias=qlinear.bias is not None,
        device=qlinear.qweight.device,
        dtype=qlinear.scales.dtype,
    )
    linear.bias = qlinear.bias

    # shapes are infeatures x outfeatures, dtypes is int8
    iweight, izeros = unpack_4bit_to_32bit_signed(qlinear.qweight, qlinear.qzeros)
    # index using g_idx to get the correct scales and zeros
    # should we use repeat_interleave is g_idx is trivial?
    scales = qlinear.scales[qlinear.g_idx]
    scales_zeros = (izeros * qlinear.scales)[qlinear.g_idx]
    # dequantize, linear.weight is outfeatures x infeatures
    dequantized_weight = (iweight * scales - scales_zeros).t()
    linear.weight.copy_(dequantized_weight.contiguous())

    return linear


@torch.no_grad()
def dequantize_auto_awq_qlinear(qlinear) -> torch.nn.Linear:
    from awq.utils.packing_utils import dequantize_gemm

    linear = torch.nn.Linear(
        qlinear.in_features,
        qlinear.out_features,
        bias=qlinear.bias is not None,
        device=qlinear.weight.device,
        dtype=qlinear.scales.dtype,
    )
    linear.bias = qlinear.bias

    dequantized_weight = dequantize_gemm(
        qlinear.qweight, qlinear.qzeros, qlinear.scales, qlinear.w_bit, qlinear.group_size
    ).t()
    linear.weight.data.copy_(dequantized_weight.contiguous())

    return linear


@torch.no_grad()
def dequantize_bnb_qlinear(qlinear) -> torch.nn.Linear:
    from bitsandbytes.functional import dequantize_4bit

    linear = torch.nn.Linear(
        qlinear.in_features,
        qlinear.out_features,
        bias=qlinear.bias is not None,
        device=qlinear.weight.device,
        dtype=qlinear.quant_state.dtype,
    )
    linear.bias = qlinear.bias

    dequantized_weight = dequantize_4bit(qlinear.weight, qlinear.quant_state).t()
    linear.weight.data.copy_(dequantized_weight.contiguous())

    return linear


DEQUANTIZE_MAPPING = {
    "awq": (get_auto_awq_qlinear_cls, dequantize_auto_awq_qlinear),
    "bitsandbytes": (get_bnb_qlinear_cls, dequantize_bnb_qlinear),
    "gptq": (get_auto_gptq_qlinear_cls, dequantize_auto_gptq_qlinear),
}


def maybe_dequantize_model(model: torch.nn.Module) -> torch.nn.Module:
    """Dequantize the model if it was quantized using one of the supported methods."""
    model, modified = _replace_qlinear_modules(model, DEQUANTIZE_MAPPING)

    if modified:
        del model.quantization_method
        del model.config.quantization_config
        if hasattr(model, "is_quantized"):
            del model.is_quantized

    return model


# Torch export compatible


# Should we also support QDQ export? Need different packing and symbolic for that
class QuantLinearTorchFunction(torch.autograd.Function):
    """Used to export the quantized linear layer to onnx using the contrib operator MatMulNBits."""

    # pylint: disable=W0223,W0221
    @staticmethod
    def symbolic(g, x, qweight, scales, qzeros, g_idx, bits, group_size, in_features, out_features):
        tensor_args = [x, qweight, scales, qzeros]
        if g_idx is not None:
            tensor_args.append(g_idx)
        attrs = {
            "K_i": in_features,
            "N_i": out_features,
            "bits_i": bits,
            "block_size_i": group_size,
        }

        output = g.op(
            "com.microsoft::MatMulNBits",
            *tensor_args,
            # what does this outputs do?
            outputs=1,
            **attrs,
        )
        input_shape = x.type().varyingSizes()
        if input_shape is not None and hasattr(x.type(), "with_sizes"):
            output_type = x.type().with_sizes(input_shape[:-1] + [qweight.type().varyingSizes()[0]])
            output.setType(output_type)

        return output

    @staticmethod
    def forward(ctx, x, qweight, scales, qzeros, g_idx, bits, group_size, in_features, out_features):
        if torch.onnx.is_in_onnx_export():
            return torch.zeros(x.shape[:-1] + (out_features,), dtype=x.dtype, device=x.device)
        raise NotImplementedError("QuantLinearTorchFunction forward is only implemented for onnx export")


class QuantLinear4bit(torch.nn.Module):
    """Quantized linear layer with 4 bits per element.

    Packing is done in the same way as MatMulNBits operator in onnxruntime.
    """

    def __init__(
        self,
        group_size: int,
        in_features: int,
        out_features: int,
        g_idx: bool = False,
        bias: bool = False,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = group_size

        # only support 4 bits for now
        bits = 4

        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, in_features // self.group_size, self.group_size // (8 // bits)), dtype=torch.uint8
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros((math.ceil(in_features // self.group_size) * (out_features // 8 * bits)), dtype=torch.uint8),
        )
        self.register_buffer(
            "scales", torch.zeros((math.ceil(in_features / self.group_size) * out_features), dtype=dtype)
        )
        if g_idx:
            self.register_buffer(
                "g_idx", torch.tensor([i // self.group_size for i in range(in_features)], dtype=torch.int32)
            )
        else:
            self.g_idx = None
        if bias:
            self.register_buffer("bias", torch.zeros((out_features), dtype=dtype))
        else:
            self.bias = None

    def forward(self, x):
        out = QuantLinearTorchFunction.apply(
            x,
            self.qweight,
            self.scales,
            self.qzeros,
            self.g_idx,
            4,
            self.group_size,
            self.in_features,
            self.out_features,
        )
        return out + self.bias if self.bias is not None else out

    def pack(self, iweight, izeros, scales, g_idx=None):
        """Pack int8 weight and zeros to int4 and int8 respectively.

        iweight, izeros and scales must have out_features as the first dimension
        iweight, izeros must be uint8 tensors with each holding 4bit values
        """
        # pylint: disable=W0201
        # shapes for packing
        n, k = iweight.shape
        block_size = self.group_size
        blob_size = block_size // 2
        k_blocks = (k + block_size - 1) // block_size

        padded_k = k_blocks * block_size
        pad_len = padded_k - k
        if pad_len > 0:
            iweight = torch.nn.functional.pad(iweight, (0, pad_len), value=0)

        # pack the weight
        packed_weight = (iweight[:, 0::2] & 0xF) | ((iweight[:, 1::2] & 0xF) << 4)
        self.qweight = packed_weight.reshape(n, k_blocks, blob_size).contiguous()

        # pad to make the K dimension even
        izeros = torch.nn.functional.pad(izeros, (0, izeros.shape[-1] & 1), value=0)
        # pack the zeros
        packed_zeros = (izeros[:, 0::2] & 0xF) | ((izeros[:, 1::2] & 0xF) << 4)
        self.qzeros = packed_zeros.flatten().contiguous()

        self.scales = scales.flatten().contiguous()

        self.g_idx = g_idx


def make_auto_awq_qlinear4bit(qlinear):
    if qlinear.w_bit != 4:
        return qlinear

    from awq.utils.packing_utils import reverse_awq_order, unpack_awq

    # shapes are KxN, dtypes is int8
    iweight, izeros = unpack_awq(qlinear.qweight, qlinear.qzeros, qlinear.w_bit)
    iweight, izeros = reverse_awq_order(iweight, izeros, qlinear.w_bit)
    # expected shape and dtype for QuanLinear
    iweight = iweight.to(torch.uint8).t()
    izeros = izeros.to(torch.uint8).t()
    scales = qlinear.scales.t()

    new_qlinear = QuantLinear4bit(
        qlinear.group_size,
        iweight.shape[1],
        iweight.shape[0],
        bias=qlinear.bias is not None,
        dtype=qlinear.scales.dtype,
    )
    new_qlinear.pack(iweight, izeros, scales)

    return new_qlinear


def make_auto_gptq_qlinear4bit(qlinear):
    if qlinear.bits != 4:
        return qlinear

    from auto_gptq.nn_modules.qlinear.qlinear_marlin import unpack_4bit_to_32bit_signed

    # shapes are KxN, dtypes is int8
    iweight, izeros = unpack_4bit_to_32bit_signed(qlinear.qweight, qlinear.qzeros)
    # expected shape and dtype for QuanLinear
    iweight = iweight.to(torch.uint8).t()
    izeros = izeros.to(torch.uint8).t()
    scales = qlinear.scales.t()
    # no need to use g_idx if descriptive activation is not used
    g_idx_trivial = torch.arange(iweight.shape[1], device=qlinear.g_idx.device, dtype=torch.int32) // qlinear.group_size
    g_idx = qlinear.g_idx if not torch.equal(qlinear.g_idx, g_idx_trivial) else None

    new_qlinear = QuantLinear4bit(
        qlinear.group_size,
        iweight.shape[1],
        iweight.shape[0],
        g_idx=g_idx is not None,
        bias=qlinear.bias is not None,
        dtype=qlinear.scales.dtype,
    )
    new_qlinear.pack(iweight, izeros, scales, g_idx)

    return new_qlinear


EXPORT_QLINEAR_MAPPING = {
    "awq": (get_auto_awq_qlinear_cls, make_auto_awq_qlinear4bit),
    "gptq": (get_auto_gptq_qlinear_cls, make_auto_gptq_qlinear4bit),
}


def make_export_compatible(model: torch.nn.Module) -> torch.nn.Module:
    """Make the model export compatible by replacing the quantized linear layers with 4-bit versions."""
    return _replace_qlinear_modules(model, EXPORT_QLINEAR_MAPPING)[0]
