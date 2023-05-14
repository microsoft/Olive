# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from olive.evaluator.metric import Metric
from olive.evaluator.metric_config import MetricResult
from olive.evaluator.olive_evaluator import OliveEvaluator, OliveEvaluatorFactory
from olive.model import CompositeOnnxModel, OliveModel
from olive.passes.olive_pass import Pass
from olive.systems.common import Device, SystemType
from olive.systems.olive_system import OliveSystem


class LocalSystem(OliveSystem):
    system_type = SystemType.Local

    def __init__(self, device: Device = Device.CPU):
        self.device = device
        super().__init__()

    def run_pass(
        self,
        the_pass: Pass,
        model: OliveModel,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> OliveModel:
        """
        Run the pass on the model at a specific point in the search space.
        """
        return the_pass.run(model, output_model_path, point)

    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> MetricResult:
        """
        Evaluate the model
        """
        if isinstance(model, CompositeOnnxModel):
            raise NotImplementedError()

        evaluator: OliveEvaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        return evaluator.evaluate(model, metrics, device=self.device)
