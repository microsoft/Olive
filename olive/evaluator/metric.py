# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict, List, Optional, Union

from olive.common.config_utils import ConfigBase, NestedConfig, validate_config
from olive.common.pydantic_v1 import validator
from olive.common.utils import StrEnumBase
from olive.data.config import DataConfig
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_config import LatencyMetricConfig, MetricGoal, ThroughputMetricConfig, get_user_config_class

logger = logging.getLogger(__name__)


class MetricType(StrEnumBase):
    # TODO(trajep): support throughput
    ACCURACY = "accuracy"
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    CUSTOM = "custom"


class AccuracySubType(StrEnumBase):
    ACCURACY_SCORE = "accuracy_score"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUROC = "auroc"
    PERPLEXITY = "perplexity"


class LatencySubType(StrEnumBase):
    # unit: millisecond
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    P999 = "p999"


class ThroughputSubType(StrEnumBase):
    # unit: token per second, tps
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    P999 = "p999"


class SubMetric(ConfigBase):
    name: Union[AccuracySubType, LatencyMetricConfig, str]
    metric_config: ConfigBase = None
    # -1 means no priority which will be evaluated only
    priority: int = -1
    higher_is_better: bool = False
    goal: MetricGoal = None

    @validator("goal")
    def validate_goal(cls, v, values):
        if v is None:
            return v
        if v.type not in ["percent-min-improvement", "percent-max-degradation"]:
            return v

        if "higher_is_better" not in values:
            raise ValueError("Invalid higher_is_better")
        higher_is_better = values["higher_is_better"]

        ranges = {
            ("percent-min-improvement", True): (0, float("inf")),
            ("percent-min-improvement", False): (0, 100),
            ("percent-max-degradation", True): (0, 100),
            ("percent-max-degradation", False): (0, float("inf")),
        }
        valid_range = ranges[(v.type, higher_is_better)]
        if not valid_range[0] < v.value < valid_range[1]:
            raise ValueError(
                f"Invalid goal value {v.value} for {v.type} and higher_is_better={higher_is_better}. Valid range is"
                f" {valid_range}"
            )
        return v


class Metric(NestedConfig):
    _nested_field_name = "user_config"

    name: str
    type: MetricType
    backend: Optional[str] = "torch_metrics"
    sub_types: List[SubMetric]
    user_config: ConfigBase = None
    data_config: Optional[DataConfig] = None

    def get_inference_settings(self, framework):
        if self.user_config is None:
            return None
        if self.user_config.inference_settings:
            return self.user_config.inference_settings.get(framework)
        else:
            return None

    def get_run_kwargs(self) -> Dict[str, Any]:
        return self.user_config.run_kwargs if (self.user_config and self.user_config.run_kwargs) else {}

    def get_sub_type_info(self, info_name, no_priority_filter=True, callback=lambda x: x):
        sub_type_info = {}
        for sub_type in self.sub_types:
            if no_priority_filter and sub_type.priority <= 0:
                continue
            sub_type_info[sub_type.name] = callback(getattr(sub_type, info_name))
        return sub_type_info

    @validator("backend", always=True, pre=True)
    def validate_backend(cls, v, values):
        if values["type"] == MetricType.CUSTOM:
            return None
        from olive.evaluator.metric_backend import MetricBackend

        assert v in MetricBackend.registry, f"Backend {v} is not in {list(MetricBackend.registry.keys())}"
        assert MetricBackend.registry[v]() is not None, f"Backend {v} is not available"
        return v

    @validator("sub_types", always=True, pre=True, each_item=True)
    def validate_sub_types(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        if values["type"] == MetricType.CUSTOM:
            if v.get("priority", -1) != -1 and v.get("higher_is_better", None) is None:
                raise ValueError(f"higher_is_better must be specified for ranked custom metric: {v['name']}")
            return v

        # backend joint checking
        if values["backend"] == "huggingface_metrics":
            import evaluate

            try:
                evaluate.load(v["name"])
            except FileNotFoundError as e:
                raise ValueError(f"could not load metric {v['name']} from huggingface/evaluate") from e
        elif values["backend"] == "torch_metrics":
            try:
                sub_metric_type_cls = None
                if values["type"] == MetricType.ACCURACY:
                    sub_metric_type_cls = AccuracySubType
                elif values["type"] == MetricType.LATENCY:
                    sub_metric_type_cls = LatencySubType
                elif values["type"] == MetricType.THROUGHPUT:
                    sub_metric_type_cls = ThroughputSubType
                # if not exist, will raise ValueError
                v["name"] = sub_metric_type_cls(v["name"])
            except ValueError:
                raise ValueError(
                    f"sub_type {v['name']} is not in {list(sub_metric_type_cls.__members__.keys())}"
                    f" for {values['type']} metric"
                ) from None

        # metric_config
        metric_config_cls = None
        if values["type"] == MetricType.ACCURACY:
            v["higher_is_better"] = v.get("higher_is_better", True)
            if values["backend"] == "torch_metrics":
                metric_config_cls = AccuracyBase.registry[v["name"]].get_config_class()
            elif values["backend"] == "huggingface_metrics":
                from olive.evaluator.metric_backend import HuggingfaceMetrics

                metric_config_cls = HuggingfaceMetrics.get_config_class()
        elif values["type"] == MetricType.LATENCY:
            v["higher_is_better"] = v.get("higher_is_better", False)
            metric_config_cls = LatencyMetricConfig
        elif values["type"] == MetricType.THROUGHPUT:
            v["higher_is_better"] = v.get("higher_is_better", True)
            metric_config_cls = ThroughputMetricConfig
        v["metric_config"] = validate_config(v.get("metric_config", {}), metric_config_cls)

        return v

    @validator("user_config", pre=True, always=True)
    def validate_user_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        user_config_class = get_user_config_class(values["type"])
        return validate_config(v, user_config_class)


def get_latency_config_from_metric(metric: Metric):
    warmup_num, repeat_test_num, sleep_num = None, None, None
    for sub_type in metric.sub_types:
        if sub_type.metric_config:
            warmup_num = sub_type.metric_config.warmup_num
            repeat_test_num = sub_type.metric_config.repeat_test_num
            sleep_num = sub_type.metric_config.sleep_num
            break
    return warmup_num, repeat_test_num, sleep_num
