# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import TYPE_CHECKING, Dict, List, NamedTuple

from olive.common.utils import get_nested_dict_value
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_result import MetricResult, SubMetricResult

if TYPE_CHECKING:
    from olive.evaluator.metric import SubMetric


class MetricBackend:
    def __init__(self, sub_types: List["SubMetric"]):
        self.sub_types = sub_types

    @abstractmethod
    def update(self, model_output: NamedTuple, targets):
        raise NotImplementedError

    @abstractmethod
    def measure(self) -> MetricResult:
        raise NotImplementedError

    def parse_result(self, result: Dict[str, float]) -> MetricResult:
        sub_type_map = {sub_type.name: sub_type for sub_type in self.sub_types}
        return MetricResult(
            {
                metric_name: SubMetricResult(
                    value=result[metric_name],
                    priority=sub_type_map[metric_name].priority,
                    higher_is_better=sub_type_map[metric_name].higher_is_better,
                )
                for metric_name in result
            }
        )


class TorchMetrics(MetricBackend):
    def __init__(self, sub_types: List["SubMetric"]):
        super().__init__(sub_types)
        self.metrics = [
            AccuracyBase.registry[sub_metric.name.value](sub_metric.metric_config) for sub_metric in sub_types
        ]

    def update(self, model_output: NamedTuple, targets):
        for metric in self.metrics:
            metric.update(model_output, targets)

    def measure(self) -> MetricResult:
        return self.parse_result(
            {sub_metric.name: metric.compute() for sub_metric, metric in zip(self.sub_types, self.metrics)}
        )


class HuggingfaceMetrics(MetricBackend):
    def __init__(self, sub_types: List["SubMetric"]):
        try:
            import evaluate
        except ImportError:
            raise ImportError("Please install the huggingface/evaluate package to use huggingface metrics.") from None

        super().__init__(sub_types)
        # pylint: disable=not-a-mapping
        self.metrics = [
            evaluate.load(sub_metric.name, **{sub_metric.metric_config.load_params or {}}) for sub_metric in sub_types
        ]

    def update(self, model_output: NamedTuple, targets):
        for sub_metric, metric in zip(self.sub_types, self.metrics):
            metric.add_batch(
                predictions=model_output.preds, references=targets, **(sub_metric.metric_config.compute_params or {})
            )

    def measure(self) -> MetricResult:
        metric_results_dict = {}
        for sub_metric, metric in zip(self.sub_types, self.metrics):
            result = metric.compute()
            if sub_metric.metric_config.result_key:
                result = get_nested_dict_value(result, sub_metric.metric_config.result_key)
            else:
                result = result[sub_metric.name]
            metric_results_dict[sub_metric.name] = result

        return self.parse_result(metric_results_dict)


backend_map = {
    "torch": TorchMetrics,
    "huggingface": HuggingfaceMetrics,
}


def is_valid_backend(backend_type: str) -> bool:
    return backend_type in backend_map


def create_metric_backend(backend_type: str, sub_types: List["SubMetric"]) -> MetricBackend:
    if backend_type not in backend_map:
        raise ValueError(f"Unknown metric backend type: {backend_type}")
    return backend_map[backend_type](sub_types)
