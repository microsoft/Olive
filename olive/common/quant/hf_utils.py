# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch.nn as nn
from transformers.integrations import get_keys_to_not_convert
from transformers.quantizers.base import HfQuantizer
from transformers.utils.quantization_config import QuantizationConfigMixin

from olive.common.utils import StrEnumBase

if TYPE_CHECKING:
    from tqdm.auto import tqdm
    from transformers import PreTrainedModel


# older transformers expects a StrEnum and accesses .value
class OliveHfQuantizationMethod(StrEnumBase):
    """Enumeration for Olive quantization methods."""

    # Olive quantization method
    OLIVE = "olive"


# TODO(jambayk): standardize this with the default quantization settings
@dataclass
class OliveHfQuantizationOverrideConfig:
    """Configuration overrides for individual modules during Olive quantization.

    Attributes:
        bits: Override for bit width (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization for this module.
        group_size: Size of quantization group for this module.

    """

    bits: int | None = None
    symmetric: bool | None = None
    group_size: int | None = None


@dataclass
class OliveHfQuantizationConfig(QuantizationConfigMixin):
    """Configuration for Olive quantization.

    Extends Hugging Face's QuantizationConfigMixin with Olive-specific settings.

    Attributes:
        bits: Default bit width for quantization (e.g. 4, 8).
        symmetric: Whether to use symmetric quantization.
        group_size: Quantization group size.
            -1 = per-channel, 0 = per-tensor, >0 = groupwise.
        modules_to_not_convert : List of module names to exclude from quantization.
        overrides: Per-module overrides for quantization parameters.

    """

    # pylint: disable
    def __init__(
        self,
        bits: int,
        symmetric: bool,
        group_size: int,
        modules_to_not_convert: list | None = None,
        overrides: dict | None = None,
        **kwargs,
    ):
        # pylint: disable=W0231
        self.quant_method = OliveHfQuantizationMethod.OLIVE

        self.bits = bits
        self.symmetric = symmetric
        self.group_size = group_size
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
            "group_size": self.group_size,
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
        from olive.common.quant.linear import QuantLinear

        # this helps skip modules such as lm_head which is generally not quantized
        # newer transformers versions have a `get_keys_to_not_convert` function
        # TODO(jambayk): maybe add an option to skip/include lm head if we start quantizing it
        self.modules_to_not_convert = get_keys_to_not_convert(model)
        if self.quantization_config.modules_to_not_convert:
            self.modules_to_not_convert.extend(self.quantization_config.modules_to_not_convert)
        if keep_in_fp32_modules:
            self.modules_to_not_convert.extend(keep_in_fp32_modules)

        def should_quantize(module: nn.Module, name: str) -> bool:
            """Check if a module should be quantized."""
            return isinstance(module, nn.Linear) and not any(key in name for key in self.modules_to_not_convert)

        def create_quantized_module(module: nn.Linear, name: str) -> QuantLinear:
            """Create a quantized version of a module."""
            return QuantLinear(
                in_features=module.in_features,
                out_features=module.out_features,
                **self.quantization_config.get_qlinear_init_args(name),
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )

        replace_matching_submodules(model, should_quantize, create_quantized_module)

    def _process_model_after_weight_loading(self, model: PreTrainedModel, **kwargs):
        return model

    def is_serializable(self, safe_serialization=None) -> bool:
        return True

    @property
    def is_trainable(self) -> bool:
        # TODO(jambayk): investigate what this means (peft, scale+bias, etc.?)
        # need to support peft, scale+bias comes for free since everything is in torch
        return False


def replace_matching_submodules(
    module: nn.Module,
    condition: Callable[[nn.Module, str], bool],
    transform: Callable[[nn.Module], nn.Module],
    path: list[str] | None = None,
    description: str | None = None,
    pbar: tqdm | None = None,
):
    """Walk the module tree and replace every sub-module that meets a condition.

    Args:
        module: Root ``nn.Module`` to start from.
        condition: ``condition(m, name) -> bool``. Return ``True`` to trigger
            replacement.
        transform: ``transform(m, name) -> nn.Module``. Produces the replacement.
        path: (Internal) List of name segments used to build the dotted path.
        description: (Internal) Description for the progress bar.
        pbar: (Internal) Progress bar to update. If ``None``, a new
            ``tqdm`` progress bar will be created.

    Returns:
        The root module with all replacements applied. If the root matches,
        the returned object is whatever ``transform`` returns.

    """
    is_root = pbar is None
    if is_root:
        from tqdm.auto import tqdm

        # TODO(jambayk): check logging level and only show progress bar if debug
        pbar = tqdm(total=None, desc=description or "Replacing submodules", unit="mod")
    path = [] if path is None else path

    name = ".".join(path)
    if condition(module, name):
        pbar.set_postfix(module=name, refresh=False)
        pbar.update(1)
        # do we need to recurse into children first? no use case for that yet
        return transform(module, name)

    for name, child in module.named_children():
        new_child = replace_matching_submodules(child, condition, transform, [*path, name], description, pbar)
        module.add_module(name, new_child)

    if is_root:
        pbar.close()

    return module
