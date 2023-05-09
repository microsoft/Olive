# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from olive.evaluator.evaluation import evaluator_adaptor
from olive.evaluator.metric import Metric
from olive.evaluator.metric_config import SignalResult
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

    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> SignalResult:
        """
        Evaluate the model
        """
        if isinstance(model, CompositeOnnxModel):
            raise NotImplementedError()

        metrics_res = {}
        for metric in metrics:
            evaluator = evaluator_adaptor(metric)
            metrics_res[metric.name] = evaluator(model, metric, self.device)
        return SignalResult(signal=metrics_res)
