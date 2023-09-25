# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from olive.evaluator.metric import Metric, MetricResult
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorFactory
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.passes.olive_pass import Pass
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def __init__(self, accelerators: List[str] = None):
        super().__init__(accelerators=accelerators, olive_managed_env=False)

    def run_pass(
        self,
        the_pass: Pass,
        model_config: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """Run the pass on the model at a specific point in the search space."""
        model = model_config.create_model()
        output_model = the_pass.run(model, data_root, output_model_path, point)
        return ModelConfig.from_json(output_model.to_json())

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> MetricResult:
        """Evaluate the model."""
        if model_config.type.lower() == "CompositeOnnxModel".lower():
            raise NotImplementedError

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        model = model_config.create_model()
        evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        return evaluator.evaluate(model, data_root, metrics, device=device, execution_providers=execution_providers)

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        import onnxruntime as ort

        return ort.get_available_providers()

    def remove(self):
        raise ValueError("Local system does not support system removal")
