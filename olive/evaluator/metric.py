# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from typing import List, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.data.config import DataConfig
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_config import LatencyMetricConfig, MetricGoal, get_user_config_class

logger = logging.getLogger(__name__)


class MetricType(str, Enum):
    ACCURACY = "accuracy"
    LATENCY = "latency"
    CUSTOM = "custom"


class AccuracySubType(str, Enum):
    ACCURACY_SCORE = "accuracy_score"
    F1_SCORE = "f1_score"
    PRECISION = "precision"
    RECALL = "recall"
    AUC = "auc"


class LatencySubType(str, Enum):
    AVG = "avg"
    MAX = "max"
    MIN = "min"
    P50 = "p50"
    P75 = "p75"
    P90 = "p90"
    P95 = "p95"
    P99 = "p99"
    P999 = "p999"


class MetricSubType(ConfigBase):
    name: Union[AccuracySubType, LatencyMetricConfig, str]
    metric_config: ConfigBase = None
    # -1 means no priority which will be evaluated only
    priority_rank: int = -1
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


class Metric(ConfigBase):
    name: str
    type: MetricType
    sub_types: List[MetricSubType]
    user_config: ConfigBase = None
    data_config: DataConfig = DataConfig()

    def get_sub_type_info(self, info_name, no_priority_rank_filter=True, callback=lambda x: x):
        sub_type_info = {}
        for sub_type in self.sub_types:
            if no_priority_rank_filter and sub_type.priority_rank <= 0:
                continue
            sub_type_info[sub_type.name] = callback(getattr(sub_type, info_name))
        return sub_type_info

    @validator("sub_types", always=True, pre=True, each_item=True)
    def validate_sub_types(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        if values["type"] == MetricType.CUSTOM:
            if v.get("priority_rank", -1) != -1 and v.get("higher_is_better", None) is None:
                raise ValueError(f"higher_is_better must be specified for ranked custom metric: {v['name']}")
            return v
        # name
        sub_type_enum = AccuracySubType if values["type"] == MetricType.ACCURACY else LatencySubType
        try:
            v["name"] = sub_type_enum(v["name"])
        except ValueError:
            raise ValueError(
                f"sub_type {v['name']} is not in {list(sub_type_enum.__members__.keys())} for {values['type']} metric"
            )

        # metric_config
        metric_config_cls = None
        if sub_type_enum is AccuracySubType:
            v["higher_is_better"] = True
            metric_config_cls = AccuracyBase.registry[v["name"]].get_config_class()
        elif sub_type_enum is LatencySubType:
            v["higher_is_better"] = False
            metric_config_cls = LatencyMetricConfig
        v["metric_config"] = validate_config(v.get("metric_config", {}), ConfigBase, metric_config_cls)

        return v

    @validator("user_config", pre=True)
    def validate_user_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        user_config_class = get_user_config_class(values["type"])
        return validate_config(v, ConfigBase, user_config_class)
