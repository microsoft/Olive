# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorFactory
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric, MetricResult
    from olive.passes.olive_pass import Pass

logger = logging.getLogger(__name__)


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
        enable_fast_mode: bool = False,
    ) -> ModelConfig:
        """Run the pass on the model at a specific point in the search space."""
        model = model_config.create_model()
        model.model = model_config.loaded_model

        output_model = the_pass.run(model, data_root, output_model_path, point, enable_fast_mode)
        model_config = ModelConfig.from_json(output_model.to_json())
        model_config.loaded_model = output_model.model

        return model_config

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List["Metric"], accelerator: AcceleratorSpec
    ) -> "MetricResult":
        """Evaluate the model."""
        if model_config.type.lower() in ("compositemodel", "compositepytorchmodel"):
            raise NotImplementedError

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        model = model_config.create_model()
        model.model = model_config.loaded_model
        evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        return evaluator.evaluate(model, data_root, metrics, device=device, execution_providers=execution_providers)

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        import onnxruntime as ort

        return ort.get_available_providers()

    def remove(self):
        raise NotImplementedError("Local system does not support system removal")
