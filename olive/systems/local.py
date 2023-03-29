# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, List, Optional

from olive.evaluator.evaluation import evaluate_accuracy, evaluate_custom_metric, evaluate_latency
from olive.evaluator.metric import Metric, MetricType
from olive.model import OliveModel
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

    def evaluate_model(self, model: OliveModel, metrics: List[Metric]) -> Dict[str, Any]:
        """
        Evaluate the model
        """
        metrics_res = {}
        for metric in metrics:
            if metric.type == MetricType.ACCURACY:
                metrics_res[metric.name] = evaluate_accuracy(model, metric, self.device)
            elif metric.type == MetricType.LATENCY:
                metrics_res[metric.name] = evaluate_latency(model, metric, self.device)
            else:
                metrics_res[metric.name] = evaluate_custom_metric(model, metric, self.device)
        return metrics_res
