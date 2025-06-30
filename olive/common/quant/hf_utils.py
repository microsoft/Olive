# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch.nn as nn
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

if TYPE_CHECKING:
    from transformers import PreTrainedModel


QUANTIZATION_METHOD = "olive"


# TODO(jambayk): standardize this with the default quantization settings
@dataclass
class OliveHfQuantizationOverrideConfig:
    """Configuration overrides for individual modules during Olive quantization.

    Attributes:
        bits: Override for bit width (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization for this module.
        block_size: Size of quantization block for this module.

    """

    bits: int | None = None
    symmetric: bool | None = None
    block_size: int | None = None


@dataclass
class OliveHfQuantizationConfig(QuantizationConfigMixin):
    """Configuration for Olive quantization.

    Extends Hugging Face's QuantizationConfigMixin with Olive-specific settings.

    Attributes:
        bits: Default bit width for quantization (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization.
        block_size: Quantization block size.
            -1 = per-channel, 0 = per-tensor, >0 = blockwise.
        modules_to_not_convert : List of module names to exclude from quantization.
        overrides: Per-module overrides for quantization parameters.

    """

    # pylint: disable
    def __init__(
        self,
        bits: int,
        symmetric: bool,
        block_size: int,
        modules_to_not_convert: list | None = None,
        overrides: dict | None = None,
        **kwargs,
    ):
        # pylint: disable=W0231
        self.quant_method = QUANTIZATION_METHOD

        self.bits = bits
        self.symmetric = symmetric
        self.block_size = block_size
        self.modules_to_not_convert = modules_to_not_convert
        self.overrides = {
            module_name: OliveHfQuantizationOverrideConfig(**override)
            for module_name, override in (overrides or {}).items()
        }
        self.post_init()

    def post_init(self):
        """Safety checker that arguments are correct."""
        if self.bits not in [4, 8]:
            raise ValueError(f"Only 4-bit and 8-bit quantization supported, got {self.bits}")

    def to_dict(self) -> dict:
        """Serialize this instance to a Python dictionary."""
        output = super().to_dict()
        if self.overrides:
            overrides = {}
            for module_name, override in self.overrides.items():
                # remove None or default values from the override
                cleaned_override = {k: v for k, v in override.__dict__.items() if v is not None and v != output[k]}
                if cleaned_override:
                    overrides[module_name] = cleaned_override
            output["overrides"] = overrides or None
        else:
            output["overrides"] = None
        return output

    def get_qlinear_init_args(self, module_name: str) -> dict:
        """Get the initialization arguments for a QuantLinear layer based on the module name.

        Args:
            module_name (str): The name of the module to get initialization args for.

        Returns:
            dict: Initialization arguments for QuantLinear.

        """
        init_args = {
            "bits": self.bits,
            "symmetric": self.symmetric,
            "block_size": self.block_size,
        }
        if override := self.overrides.get(module_name):
            init_args.update({k: v for k, v in override.__dict__.items() if v is not None})
        return init_args


class OliveHfQuantizer(HfQuantizer):
    """Olive quantizer."""

    # only support load and inference, no on-the-fly quantization
    requires_calibration = True

    def _process_model_before_weight_loading(
        self, model: PreTrainedModel, keep_in_fp32_modules: list[str] | None = None, **kwargs
    ):
        # this helps skip modules such as lm_head which is generally not quantized
        # TODO(jambayk): maybe add an option to skip/include lm head if we start quantizing it
        self.modules_to_not_convert = self.get_modules_to_not_convert(
            model, self.quantization_config.modules_to_not_convert, keep_in_fp32_modules, add_default_skips=True
        )

        model, _ = replace_with_quant_linear(
            model, quantization_config=self.quantization_config, modules_to_not_convert=self.modules_to_not_convert
        )

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        # TODO(jambayk): investigate what this means (peft, scale+bias, etc.?)
        # need to support peft, scale+bias comes for free since everything is in torch
        return False


def replace_with_quant_linear(
    model: nn.Module,
    quantization_config: OliveHfQuantizationConfig,
    modules_to_not_convert: list[str] | None = None,
    current_key_name: list[str] | None = None,
    has_been_replaced: bool = False,
) -> bool:
    """Recursively replace the Linear layers of the given model with Olive quantized layers.

    Args:
        model: The model to convert, can be any `torch.nn.Module` instance.
        quantization_config: The quantization config object that contains the quantization parameters.
        modules_to_not_convert: A list of modules to not convert. If a module name is in the list (e.g. `lm_head`), it will not be
            converted.
        current_key_name: A list that contains the current key name. This is used for recursion and should not be passed by the user.
        has_been_replaced: A boolean that indicates if the conversion has been successful or not. This is used for recursion and
            should not be passed by the user.

    Returns:
        A tuple containing the converted model and a boolean that indicates if the conversion has been successful or not.

    """
    from olive.common.quant.linear import QuantLinear

    if modules_to_not_convert is None:
        modules_to_not_convert = []

    for name, module in model.named_children():
        if current_key_name is None:
            current_key_name = []
        current_key_name.append(name)

        if isinstance(module, nn.Linear) and name not in modules_to_not_convert:
            # Check if the current key is not in the `modules_to_not_convert`
            full_name = ".".join(current_key_name)
            if not any(key in full_name for key in modules_to_not_convert):
                # pylint: disable=W0212
                model._modules[name] = QuantLinear(
                    in_features=module.in_features,
                    out_features=module.out_features,
                    **quantization_config.get_qlinear_init_args(full_name),
                    bias=module.bias is not None,
                    device=module.weight.device,
                    dtype=module.weight.dtype,
                )

                has_been_replaced = True

                # Force requires grad to False to avoid unexpected errors
                model._modules[name].requires_grad_(False)
        if len(list(module.children())) > 0:
            _, has_been_replaced = replace_with_quant_linear(
                module,
                modules_to_not_convert=modules_to_not_convert,
                current_key_name=current_key_name,
                quantization_config=quantization_config,
                has_been_replaced=has_been_replaced,
            )
        # Remove the last key for recursion
        current_key_name.pop(-1)
    return model, has_been_replaced
