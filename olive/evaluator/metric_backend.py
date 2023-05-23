# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import Any, Dict, Union

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric import Metric, MetricResult, SubMetric, SubMetricResult


class MetricBackend(AutoConfigClass):
    registry: Dict[str, "MetricBackend"] = {}

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None) -> None:
        super().__init__(config)

    @staticmethod
    def _default_config() -> Dict[str, ConfigParam]:
        return {}

    @abstractmethod
    def measure_sub_metric(self, preds, targets, sub_metric: SubMetric) -> SubMetricResult:
        raise NotImplementedError()

    def measure(self, preds, targets, metrics: Metric) -> MetricResult:
        metric_results_dict = {}
        for sub_metric in metrics.sub_types:
            metric_results_dict[sub_metric.name] = self.measure_sub_metric(preds, targets, sub_metric)
        return MetricResult.parse_obj(metric_results_dict)


class TorchMetrics(MetricBackend):
    name: str = "torch_metrics"

    def measure_sub_metric(self, preds, targets, sub_metric: SubMetric) -> SubMetricResult:
        metric_cls = AccuracyBase.registry[sub_metric.name.value]
        metric_obj = metric_cls(sub_metric.metric_config)
        result = metric_obj.measure(preds, targets)
        return SubMetricResult(
            value=result,
            priority=sub_metric.priority,
            higher_is_better=sub_metric.higher_is_better,
        )


class HuggingfaceMetrics(MetricBackend):
    name: str = "huggingface_metrics"

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None) -> None:
        super().__init__(config)
        try:
            import evaluate
        except ImportError:
            raise ImportError("Please install the huggingface/evaluate package to use huggingface metrics.")
        self.evaluate_module = evaluate

    @staticmethod
    def _default_config() -> Dict[str, ConfigParam]:
        return {
            "load_params": ConfigParam(
                type_=Dict[str, Any], default_value=None, description="The parameters to load the metric."
            ),
            "compute_params": ConfigParam(
                type_=Dict[str, Any], default_value=None, description="The parameters to compute the metric."
            ),
            "post_lambda_str": ConfigParam(
                type_=str, default_value=None, description="The lambda function to post-process the metric."
            ),
        }

    def measure_sub_metric(self, preds, target, sub_metric: SubMetric) -> SubMetricResult:
        load_params = sub_metric.metric_config.load_params or {}
        evaluator = self.evaluate_module.load(sub_metric.name, **load_params)

        compute_params = sub_metric.metric_config.compute_params or {}
        result = evaluator.compute(predictions=preds, references=target, **compute_params)

        post_lambda_str = sub_metric.metric_config.post_lambda_str or None

        if post_lambda_str:
            result = eval(post_lambda_str)(result)
        else:
            result = result[sub_metric.name]
        return SubMetricResult(
            value=result,
            priority=sub_metric.priority,
            higher_is_better=sub_metric.higher_is_better,
        )
