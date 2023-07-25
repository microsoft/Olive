# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from olive.evaluator.metric import Metric, MetricResult
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorFactory
from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.model import CompositeOnnxModel, OliveModel
from olive.passes.olive_pass import Pass
from olive.systems.common import SystemType
from olive.systems.olive_system import OliveSystem


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def __init__(self, accelerators: List[str] = None):
        super().__init__(accelerators=accelerators)

    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        return the_pass.run(model, data_root, output_model_path, point)

    def evaluate_model(
        self, model: OliveModel, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> MetricResult:
        """
        Evaluate the model
        """
        if isinstance(model, CompositeOnnxModel):
            raise NotImplementedError()

        device = accelerator.accelerator_type if accelerator else Device.CPU
        execution_providers = accelerator.execution_provider if accelerator else None

        evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        return evaluator.evaluate(model, data_root, metrics, device=device, execution_providers=execution_providers)

    @staticmethod
    def get_supported_execution_providers():
        import onnxruntime as ort

        return ort.get_available_providers()
