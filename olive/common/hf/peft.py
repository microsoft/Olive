# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import logging

import torch

logger = logging.getLogger(__name__)


def is_peft_model(model: torch.nn.Module) -> bool:
    """Check if the model is a PeftModel."""
    if importlib.util.find_spec("peft"):
        from peft import PeftModel

        return isinstance(model, PeftModel)
    return False


@torch.no_grad()
def make_export_compatible_peft(model: torch.nn.Module, merge_weights: bool = False) -> torch.nn.Module:
    """Make PeftModel torch.onnx.export compatible.

    If the model is a PeftModel:
        - Use the base model for exporting
        - Rescale the lora_B weights with scaling factor, change scaling factor to 1 (int) so that it doesn't appear
          in the exported model (only works for torchscript export)
    """
    # if pytorch_model is PeftModel, we need to get the base model
    # otherwise, the model forward has signature (*args, **kwargs) and torch.onnx.export ignores the dummy_inputs
    if is_peft_model(model):
        if merge_weights:
            return model.merge_and_unload()
        model = model.get_base_model()

    try:
        from peft.tuners.lora import LoraLayer
    except ImportError:
        logger.debug("Peft is not installed. Skipping PeftModel compatibility.")
        return model

    for module in model.modules():
        if (
            not isinstance(module, LoraLayer)
            or len(module.active_adapters) != 1
            or getattr(module, "use_dora", {}).get(module.active_adapters[0], False)
        ):
            # these cases are complicated and not seen in normal use cases
            continue

        active_adapter = module.active_adapters[0]

        # linear or embedding
        # conv will be supported in the future if needed
        lora_B_dict = module.lora_B or module.lora_embedding_B  # noqa: N806
        if active_adapter not in lora_B_dict:
            continue
        lora_B = lora_B_dict[active_adapter]  # noqa: N806
        if not isinstance(lora_B, (torch.nn.Linear, torch.nn.Parameter)):
            continue

        # multiply the weights by the scaling factor
        lora_B.weight.data.mul_(module.scaling[active_adapter])

        # change the scaling factor to 1 so that it doesn't appear in the exported model
        module.scaling[active_adapter] = 1

    return model
