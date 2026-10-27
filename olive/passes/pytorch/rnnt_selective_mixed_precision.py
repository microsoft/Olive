# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING, Any

import torch

from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec

logger = logging.getLogger(__name__)


class RnntSelectiveMixedPrecision(Pass):
    """Annotate RNNT models with mixed-precision metadata without depending on HF transformer assumptions.

    This uses the model structure itself instead of the generic ``ModelWrapper`` contract, which
    expects transformer-centric layer names like ``model.layers`` and an HF ``config`` instance.
    RNNT models commonly expose decoder/encoder/joint subtrees and standard ``nn.Linear`` layers.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: "AcceleratorSpec") -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(type_=int, default_value=4, description="Default integer precision bits."),
            "high_bits": PassConfigParam(
                type_=int, default_value=8, description="Higher precision bits for selected RNNT submodules."
            ),
            "group_size": PassConfigParam(type_=int, default_value=64, description="Default quantization group size."),
            "high_group_size": PassConfigParam(
                type_=int, default_value=64, description="Quantization group size for selected RNNT submodules."
            ),
            "qparam_strategy": PassConfigParam(
                type_=str,
                default_value="rtn",
                description="Q-parameter strategy for RNNT weight quantization: 'rtn' or 'kquant'.",
            ),
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: "AcceleratorSpec") -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        for field_name in ("bits", "high_bits"):
            value = getattr(config, field_name)
            if value not in (2, 4, 8):
                logger.error("%s must be one of {2, 4, 8} for RNNT mixed precision.", field_name)
                return False

        for field_name in ("group_size", "high_group_size"):
            value = getattr(config, field_name)
            if value <= 0:
                logger.error("%s must be greater than 0.", field_name)
                return False

        return True

    def _run_for_config(
        self,
        model: Any,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Any:
        base_model = model.load_model(cache_model=False) if hasattr(model, "load_model") else model
        if not isinstance(base_model, torch.nn.Module):
            raise TypeError("RnntSelectiveMixedPrecision expects a torch.nn.Module or PyTorchModelHandler instance.")

        module_names = [name for name, module in base_model.named_modules() if isinstance(module, torch.nn.Linear)]
        if not module_names:
            raise ValueError("RNNT model does not contain any nn.Linear modules to annotate.")

        default = {
            "bits": int(config.bits),
            "group_size": int(config.group_size),
            "qparam_strategy": getattr(config, "qparam_strategy", "rtn"),
        }
        overrides: dict[str, dict[str, int | str]] = {}
        for name in module_names:
            if "joint" in name or "decoder" in name:
                overrides[name] = {
                    "bits": int(config.high_bits),
                    "group_size": int(config.high_group_size),
                    "qparam_strategy": getattr(config, "qparam_strategy", "rtn"),
                }
            else:
                overrides[name] = {
                    "bits": int(config.bits),
                    "group_size": int(config.group_size),
                    "qparam_strategy": getattr(config, "qparam_strategy", "rtn"),
                }

        output_model = copy.deepcopy(model)
        output_model.model_attributes = getattr(output_model, "model_attributes", None) or {}
        output_model.model_attributes["mixed_precision_info"] = {
            "default": default,
            "overrides": overrides,
        }
        return output_model
