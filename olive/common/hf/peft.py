# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import logging
from contextlib import contextmanager
from copy import deepcopy
from typing import TYPE_CHECKING

import torch
from packaging import version

from olive.common.utils import get_attr

if TYPE_CHECKING:
    from peft.tuners.lora import Linear as LoraLinear

logger = logging.getLogger(__name__)


def is_peft_model(model: torch.nn.Module) -> bool:
    """Check if the model is a PeftModel."""
    if importlib.util.find_spec("peft"):
        from peft import PeftModel

        return isinstance(model, PeftModel)
    return False


class ScaledLoraLinear(torch.nn.Module):
    """A wrapper for the LoraLinear layer that pre-multiplies the lora_B weights by the scaling factor."""

    def __init__(self, original: "LoraLinear"):
        super().__init__()
        active_adapter = original.active_adapter[0]
        self.base_layer = original.base_layer
        # using a deepcopy so that the name of node in exported model says lora_A instead of default
        # should be okay since lora linears are small
        self.lora_A = deepcopy(original.lora_A[active_adapter])
        # copy the weights and scale them by the scaling factor
        # don't want to modify the original weights
        self.lora_B = deepcopy(original.lora_B[active_adapter])
        self.lora_B.weight.data *= original.scaling[active_adapter]

    def forward(self, x, *args, **kwargs):
        previous_dtype = x.dtype

        result = self.base_layer(x, *args, **kwargs)
        x - x.to(self.lora_A.weight.dtype)
        result += self.lora_B(self.lora_A(x))

        return result.to(previous_dtype)


@contextmanager
def peft_export_context_manager(model: torch.nn.Module):
    """Context manager for handling PeftModel models.

    If the model is a PeftModel:
        - Use the base model for exporting
        - Replace all LoraLinear layers with ScaledLoraLinear layers so that the scaling factor is applied to
          the weights beforehand and doesn't appear as a separate Mul node in the exported model.
    """
    if not is_peft_model(model):
        yield model
        return

    # if pytorch_model is PeftModel, we need to get the base model
    # otherwise, the model forward has signature (*args, **kwargs) and torch.onnx.export ignores the dummy_inputs
    model = model.get_base_model()

    from peft import __version__ as peft_version

    if version.parse(peft_version) < version.parse("0.7.0"):
        logger.warning(
            "Model is a peft model but the peft version is not supported for exporting with fold_lora_scale!"
            " Please use peft version 0.7.0 or higher."
        )
        yield model
        return

    from peft.tuners.lora import Linear as LoraLinear

    original_linears = []
    for name, module in model.named_modules():
        if (
            not isinstance(module, LoraLinear)
            or len(module.active_adapters) != 1
            or getattr(module, "use_dora", {}).get(module.active_adapters[0], False)
        ):
            continue

        parent_name = ".".join(name.split(".")[:-1])
        parent_module = get_attr(model, parent_name)
        target_name = name.split(".")[-1]

        scaled_linear = ScaledLoraLinear(module)
        setattr(parent_module, target_name, scaled_linear)
        original_linears.append((parent_module, target_name, module))
    try:
        yield model
    finally:
        for parent_module, target_name, linear in original_linears:
            setattr(parent_module, target_name, linear)
