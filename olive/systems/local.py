# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, List

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
    from olive.passes.olive_pass import Pass


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def run_pass(
        self,
        the_pass: "Pass",
        model_config: ModelConfig,
        output_model_path: str,
    ) -> ModelConfig:
        """Run the pass on the model."""
        model = model_config.create_model()
        output_model = the_pass.run(model, output_model_path)
        return ModelConfig.from_json(output_model.to_json())

    def evaluate_model(
        self, model_config: ModelConfig, evaluator_config: "OliveEvaluatorConfig", accelerator: AcceleratorSpec
    ) -> "MetricResult":
        """Evaluate the model."""
        if model_config.type.lower() == "compositemodel":
            raise NotImplementedError

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        model = model_config.create_model()
        evaluator: OliveEvaluator = evaluator_config.create_evaluator(model)
        return evaluator.evaluate(
            model, evaluator_config.metrics, device=device, execution_providers=execution_providers
        )

    def get_supported_execution_providers(self) -> List[str]:
        """Get the available execution providers."""
        import onnxruntime as ort

        return ort.get_available_providers()

    def remove(self):
        raise NotImplementedError("Local system does not support system removal")
