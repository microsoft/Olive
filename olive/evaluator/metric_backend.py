# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, Dict, NamedTuple, Tuple, Type, Union

from olive.common.auto_config import AutoConfigClass, ConfigBase
from olive.common.config_utils import ConfigParam
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_result import MetricResult, SubMetricResult

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric, SubMetric


class MetricBackend(AutoConfigClass):
    registry: ClassVar[Dict[str, Type["MetricBackend"]]] = {}

    def __init__(self, config: Union[ConfigBase, Dict[str, Any]] = None) -> None:
        super().__init__(config)

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {}

    @abstractmethod
    def measure_sub_metric(
        self, model_output: Union[Tuple, NamedTuple], targets: Any, sub_metric: "SubMetric"
    ) -> SubMetricResult:
        # model_output: (preds, logits)
        raise NotImplementedError

    def measure(self, model_output, targets, metrics: "Metric") -> MetricResult:
        metric_results_dict = {}
        for sub_metric in metrics.sub_types:
            metric_results_dict[sub_metric.name] = self.measure_sub_metric(model_output, targets, sub_metric)
        return MetricResult.parse_obj(metric_results_dict)


class TorchMetrics(MetricBackend):
    name: str = "torch_metrics"

    def measure_sub_metric(self, model_output, targets, sub_metric: "SubMetric") -> SubMetricResult:
        metric_cls = AccuracyBase.registry[sub_metric.name.value]
        metric_obj = metric_cls(sub_metric.metric_config)
        result = metric_obj.measure(model_output, targets)
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
            raise ImportError("Please install the huggingface/evaluate package to use huggingface metrics.") from None
        self.evaluate_module = evaluate

    @classmethod
    def _default_config(cls) -> Dict[str, ConfigParam]:
        return {
            "load_params": ConfigParam(
                type_=Dict[str, Any], default_value=None, description="The parameters to load the metric."
            ),
            "compute_params": ConfigParam(
                type_=Dict[str, Any], default_value=None, description="The parameters to compute the metric."
            ),
            "result_key": ConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "The key used to extract the metric result with given format."
                    "For example, if the metric result is {'accuracy': {'value': 0.9}},"
                    "then the result_key should be 'accuracy.value'."
                ),
            ),
        }

    def measure_sub_metric(self, model_output, targets, sub_metric: "SubMetric") -> SubMetricResult:
        load_params = sub_metric.metric_config.load_params or {}
        evaluator = self.evaluate_module.load(sub_metric.name, **load_params)

        compute_params = sub_metric.metric_config.compute_params or {}
        result = evaluator.compute(predictions=model_output[0], references=targets, **compute_params)
        if not result:
            raise ValueError(
                f"Cannot find the result for {sub_metric.name} in the metric result. Please check your parameters."
            )

        result_key = sub_metric.metric_config.result_key or None

        if result_key:
            result_key_list = result_key.split(".")
            for k in result_key_list:
                result = result.get(k, None)
                if result is None:
                    raise ValueError(f"Cannot find the result with key {k} of {result_key} in the metric result.")
        else:
            result = result[sub_metric.name]
        return SubMetricResult(
            value=result,
            priority=sub_metric.priority,
            higher_is_better=sub_metric.higher_is_better,
        )
