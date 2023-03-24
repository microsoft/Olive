# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Dict, List

from pydantic import validator

from olive.common.config_utils import ConfigBase
from olive.evaluator.metric import Metric
from olive.model import OliveModel
from olive.systems.common import SystemType
from olive.systems.local import LocalSystem
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)


class OliveEvaluator:
    def __init__(self, metrics: List[Metric], target: OliveSystem = None):
        metric_names = set([metric.name for metric in metrics])
        assert len(metric_names) == len(metrics), "Metric names must be unique"
        self.metrics = metrics
        self.target = target or LocalSystem()

    def get_metric(self, metric_name: str) -> Metric:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"Metric {metric_name} not found")

    def evaluate(self, model: OliveModel) -> Dict:
        return self.target.evaluate_model(model, self.metrics)


class OliveEvaluatorConfig(ConfigBase):
    metrics: List[Metric]
    target: SystemConfig = SystemConfig(type=SystemType.Local)

    @validator("metrics")
    def validate_metrics(cls, v):
        metric_names = set([metric.name for metric in v])
        assert len(metric_names) == len(v), "Metric names must be unique"
        return v

    def create_evaluator(self):
        return OliveEvaluator(self.metrics, self.target.create_system())
