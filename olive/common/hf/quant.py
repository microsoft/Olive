# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
from typing import Callable, Dict, Tuple

import torch
import transformers

from olive.common.utils import get_attr, set_attr

logger = logging.getLogger(__name__)


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


# copied from https://github.com/huggingface/peft/blob/v0.13.0/src/peft/utils/other.py#L579
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


@torch.no_grad()
def _replace_qlinear_modules(
    model: torch.nn.Module, mapping: Dict[str, Tuple[Callable, Callable]], desc: str
) -> Tuple[torch.nn.Module, bool]:
    """Make the model export compatible by replacing the quantized linear layers with 4-bit versions.

    :param model: the model to make export compatible. Only modified if model is quantized using a supported method.
    :param mapping: the mapping from quantization method to
        - function to get qlinear class from quantization config
        - function to get new class to replace qlinear modules with
    :param desc: description of the operation
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
    logger.debug(desc)
    logger.debug("Quantization method: %s, QuantLinear class: %s", quantization_method, qlinear_class)
    num_modified = 0
    for name, module in model.named_modules():
        if isinstance(module, qlinear_class):
            new_module = mapping[quantization_method][1](module)
            set_attr(model, name, new_module)
            # quantized lora layers have another reference to the qlinear module
            parent = get_attr(model, ".".join(name.split(".")[:-1]))
            if parent is not module and getattr(parent, "quant_linear_module", None) is module:
                set_attr(parent, "quant_linear_module", new_module)
            num_modified += 1
    logger.debug("Modified %d modules", num_modified)

    return model, num_modified > 0


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
        n_blocks_per_col = math.ceil(in_features / self.group_size)

        self.register_buffer(
            "qweight",
            torch.zeros(
                (out_features, n_blocks_per_col, math.ceil(self.group_size * bits / 8)),
                dtype=torch.uint8,
            ),
        )
        self.register_buffer(
            "qzeros",
            torch.zeros(out_features * math.ceil(n_blocks_per_col * bits / 8), dtype=torch.uint8),
        )
        self.register_buffer("scales", torch.zeros((out_features * n_blocks_per_col), dtype=dtype))
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


# TODO(jambayk): maybe support bitsandbytes too. Need to consider dtype compatibility
EXPORT_QLINEAR_MAPPING = {
    "awq": (get_auto_awq_qlinear_cls, make_auto_awq_qlinear4bit),
    "gptq": (get_auto_gptq_qlinear_cls, make_auto_gptq_qlinear4bit),
}


def make_export_compatible_quant(model: torch.nn.Module) -> torch.nn.Module:
    """Make the model export compatible by replacing the quantized linear layers with 4-bit versions."""
    model, modified = _replace_qlinear_modules(
        model, EXPORT_QLINEAR_MAPPING, "Making export compatible quantized model"
    )
    if modified:
        # set quantization method to None, gptq doesn't allow dtype casting
        model.quantization_method = None
    return model
