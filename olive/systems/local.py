# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorFactory
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.model.handler.base import OliveModelHandler
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
        input_model: OliveModelHandler,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
        enable_fast_mode: bool = False,
    ) -> OliveModelHandler:
        """Run the pass on the model at a specific point in the search space."""
        return the_pass.run(input_model, data_root, output_model_path, point, enable_fast_mode)

    def evaluate_model(
        self, input_model: OliveModelHandler, data_root: str, metrics: List["Metric"], accelerator: AcceleratorSpec
    ) -> "MetricResult":
        """Evaluate the model."""
        if input_model.type.lower() in ("compositemodel", "compositepytorchmodel"):
            raise NotImplementedError

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        return evaluator.evaluate(
            input_model, data_root, metrics, device=device, execution_providers=execution_providers
        )

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        import onnxruntime as ort

        return ort.get_available_providers()

    def remove(self):
        raise NotImplementedError("Local system does not support system removal")
