# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Component-aware activation range calibration for multimodal models."""

from __future__ import annotations

import logging
from copy import deepcopy
from typing import TYPE_CHECKING, Optional, Union

import torch

from olive.common.utils import tensor_data_to_device
from olive.data.config import DataConfig
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.multimodal_mixed_precision import Component, classify_component
from olive.passes.pytorch.multimodal_quantization import (
    ANSWER_MASK_KEY,
    VISION_MASK_KEY,
    ActivationRangeObserver,
)
from olive.passes.pytorch.train_utils import get_calibration_dataset, load_hf_base_model

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import HfModelHandler

logger = logging.getLogger(__name__)


class MultimodalActivationRangeCalibration(Pass):
    """Collect separate static activation ranges for multimodal model components.

    This is an MQuant-inspired diagnostic/planning pass, not an implementation
    of MQuant MSQ. It records component-level ranges but does not insert runtime
    modality selection, activation quantizers, or AIFS token reordering.
    """

    _SCHEMA_VERSION = 1

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "data_config": PassConfigParam(
                type_=Union[DataConfig, dict],
                default_value=None,
                description="Required processor-ready multimodal calibration data.",
            ),
            "components": PassConfigParam(
                type_=list[str],
                default_value=["vision", "audio", "text", "projector"],
                description="Components whose Linear input ranges should be observed.",
            ),
            "bits": PassConfigParam(
                type_=int,
                default_value=8,
                description="Diagnostic activation bit width used to derive qparams.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether derived diagnostic activation qparams are symmetric.",
            ),
            "component_map": PassConfigParam(
                type_=dict,
                default_value=None,
                description="Optional component-to-module-name-substrings classification overrides.",
            ),
            "device": PassConfigParam(
                type_=Optional[str],
                default_value=None,
                description="Calibration device. Defaults to CUDA when available, otherwise CPU.",
            ),
        }

    @classmethod
    def validate_config(cls, config: type[BasePassConfig], accelerator_spec: AcceleratorSpec) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False
        if config.data_config is None:
            logger.info("MultimodalActivationRangeCalibration requires data_config.")
            return False
        valid_components = {str(component) for component in Component if component != Component.OTHER}
        if unknown := set(config.components) - valid_components:
            logger.info("Unknown activation range components: %s.", sorted(unknown))
            return False
        if config.bits < 2 or config.bits > 16:
            logger.info("Activation range bits must be between 2 and 16.")
            return False
        return True

    @torch.no_grad()
    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        device = config.device or ("cuda" if torch.cuda.is_available() else "cpu")
        pytorch_model = load_hf_base_model(model, torch_dtype=model.get_load_kwargs().get("torch_dtype") or "auto")
        pytorch_model.eval().to(device)

        selected_components = set(config.components)
        extra_rules = [(substrings, component) for component, substrings in (config.component_map or {}).items()]
        observers = {component: ActivationRangeObserver(config.bits, config.sym) for component in selected_components}
        module_counts = dict.fromkeys(selected_components, 0)
        handles = []

        def make_hook(component):
            def observe_input(_, args):
                if not args or not isinstance(args[0], torch.Tensor):
                    raise ValueError("Observed Linear module did not receive a tensor as its first positional input.")
                observers[component].update(args[0])

            return observe_input

        for name, module in pytorch_model.named_modules():
            if not isinstance(module, torch.nn.Linear):
                continue
            component = str(classify_component(name, extra_rules=extra_rules))
            if component not in selected_components:
                continue
            handles.append(module.register_forward_pre_hook(make_hook(component)))
            module_counts[component] += 1

        missing_components = sorted(component for component, count in module_counts.items() if count == 0)
        observed_components = selected_components - set(missing_components)
        if not observed_components:
            raise ValueError(
                f"No Linear modules were classified into requested components {sorted(selected_components)}. "
                "Adjust components or component_map."
            )
        if missing_components:
            logger.warning("Skipping requested components with no classified Linear modules: %s.", missing_components)

        original_use_cache = getattr(pytorch_model.config, "use_cache", None)
        if original_use_cache is not None:
            pytorch_model.config.use_cache = False
        try:
            for batch in get_calibration_dataset(model, config.data_config):
                model_inputs = {
                    key: value for key, value in batch.items() if key not in {VISION_MASK_KEY, ANSWER_MASK_KEY}
                }
                pytorch_model(**tensor_data_to_device(model_inputs, device))
        finally:
            for handle in handles:
                handle.remove()
            if original_use_cache is not None:
                pytorch_model.config.use_cache = original_use_cache
            pytorch_model.to("cpu")

        ranges = {component: observers[component].qparams() for component in sorted(observed_components)}
        output = deepcopy(model)
        output.model_attributes = output.model_attributes or {}
        output.model_attributes["multimodal_activation_ranges"] = {
            "schema_version": self._SCHEMA_VERSION,
            "provenance": {
                "kind": "mquant_inspired_diagnostic",
                "reference": "arXiv:2502.00425",
                "runtime_consumer": False,
                "aifs": False,
            },
            "observer": {
                "axis": "per_tensor",
                "signed": config.sym,
                "clipping": "minmax",
            },
            "component_rules": {
                "selected": sorted(selected_components),
                "observed": sorted(observed_components),
                "missing": missing_components,
                "custom": config.component_map or {},
                "linear_module_counts": module_counts,
            },
            "ranges": ranges,
        }
        return output
