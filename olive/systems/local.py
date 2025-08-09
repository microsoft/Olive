# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import TYPE_CHECKING, Any, Union

from olive.common.config_utils import validate_config
from olive.common.ort_inference import get_ort_available_providers, maybe_register_ep_libraries
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import ModelConfig
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorConfig
    from olive.passes.olive_pass import Pass


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def __init__(
        self,
        accelerators: Union[list[AcceleratorConfig], list[dict[str, Any]]] = None,
        hf_token: bool = None,
    ):
        super().__init__(accelerators, hf_token)

        if accelerators:
            accelerators = [validate_config(accelerator, AcceleratorConfig) for accelerator in accelerators]

            maybe_register_ep_libraries(
                {name: path for accelerator in accelerators for name, path in accelerator.get_ep_path_map().items()}
            )

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

    def get_supported_execution_providers(self) -> list[str]:
        """Get the available execution providers."""
        return get_ort_available_providers()

    def remove(self):
        raise NotImplementedError("Local system does not support system removal")
