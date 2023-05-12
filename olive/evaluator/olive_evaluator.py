# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import List

from pydantic import validator

from olive.common.config_utils import ConfigBase
from olive.evaluator.metric import Metric, MetricList
from olive.evaluator.metric_config import MetricResult, joint_metric_key
from olive.model import OliveModel
from olive.systems.local import LocalSystem
from olive.systems.olive_system import OliveSystem

logger = logging.getLogger(__name__)


class OliveEvaluator:
    def __init__(self, metrics: List[Metric]):
        metric_names = set([metric.name for metric in metrics])
        assert len(metric_names) == len(metrics), "Metric names must be unique"
        self.metrics = metrics

    def get_metric(self, metric_name: str) -> Metric:
        for metric in self.metrics:
            if metric.name == metric_name:
                return metric
        raise ValueError(f"Metric {metric_name} not found")

    def evaluate(self, model: OliveModel, target: OliveSystem = LocalSystem()) -> MetricResult:
        return target.evaluate_model(model, self.metrics)


class OliveEvaluatorConfig(ConfigBase):
    metrics: MetricList

    @validator("metrics")
    def validate_metrics(cls, v):
        metric_len = len(v)
        if metric_len == 1:
            return v

        metric_names = set([metric.name for metric in v])
        assert len(metric_names) == metric_len, "Metric names must be unique"

        sub_type_names = set()
        sub_type_with_rank = set()
        rank_set = set()
        for metric in v:
            for sub_type in metric.sub_types:
                sub_type_names.add(joint_metric_key(metric.name, sub_type.name))
                if sub_type.priority_rank != -1:
                    sub_type_with_rank.add(sub_type.name)
                    rank_set.add(sub_type.priority_rank)

        if not rank_set and len(sub_type_names) == 1:
            logger.debug("""No priority rank is specified, but only one sub type
                metric is specified. Use rank 1 for single for this metric.""")
            v[0].sub_types[0].priority_rank = 1
        elif not rank_set and len(sub_type_names) > 1:
            raise ValueError("Priority rank must be specified for multiple sub type metrics")

        expected_rank_set = set(range(1, len(sub_type_with_rank) + 1))
        # Check if all ranks are present
        if rank_set != expected_rank_set:
            raise ValueError(f"Priority ranks must be unique and in the range 1 to {metric_len}")
        return v

    def create_evaluator(self):
        return OliveEvaluator(self.metrics)
