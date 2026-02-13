# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import inspect
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar, NamedTuple, Optional, Union

from olive.common.config_utils import ConfigBase
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_result import MetricResult, SubMetricResult

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric, SubMetric


class MetricBackendConfig(ConfigBase):
    """Base configuration for MetricBackend."""



class MetricBackend(ABC):
    registry: ClassVar[dict[str, type["MetricBackend"]]] = {}
    name: Optional[str] = None

    def __init__(self, config: Optional[Union[ConfigBase, dict[str, Any]]] = None) -> None:
        config = config or {}
        if isinstance(config, dict):
            config = self.get_config_class()(**config)
        self.config = config

    @classmethod
    def __init_subclass__(cls, **kwargs) -> None:
        """Register the metric backend."""
        super().__init_subclass__(**kwargs)
        if inspect.isabstract(cls):
            return
        name = cls.name if cls.name is not None else cls.__name__.lower()
        cls.registry[name] = cls

    @classmethod
    def get_config_class(cls) -> type[ConfigBase]:
        """Get the configuration class."""
        return MetricBackendConfig

    @abstractmethod
    def measure_sub_metric(
        self, model_output: Union[tuple, NamedTuple], targets: Any, sub_metric: "SubMetric"
    ) -> SubMetricResult:
        # model_output: (preds, logits)
        raise NotImplementedError

    def measure(self, model_output, targets, metrics: "Metric") -> MetricResult:
        metric_results_dict = {}
        for sub_metric in metrics.sub_types:
            metric_results_dict[sub_metric.name] = self.measure_sub_metric(model_output, targets, sub_metric)
        return MetricResult.model_validate(metric_results_dict)


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


class HuggingfaceMetricsConfig(ConfigBase):
    """Configuration for HuggingfaceMetrics."""

    load_params: Optional[dict[str, Any]] = None
    compute_params: Optional[dict[str, Any]] = None
    result_key: Optional[str] = None


class HuggingfaceMetrics(MetricBackend):
    name: str = "huggingface_metrics"

    def __init__(self, config: Optional[Union[ConfigBase, dict[str, Any]]] = None) -> None:
        config = config or {}
        if isinstance(config, dict):
            config = HuggingfaceMetricsConfig(**config)
        self.config = config
        try:
            import evaluate
        except ImportError:
            raise ImportError("Please install the huggingface/evaluate package to use huggingface metrics.") from None
        self.evaluate_module = evaluate

    @classmethod
    def get_config_class(cls) -> type[ConfigBase]:
        """Get the configuration class."""
        return HuggingfaceMetricsConfig

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
