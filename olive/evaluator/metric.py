# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from typing import List, Union

from pydantic import BaseModel, validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.data.config import DataConfig
from olive.evaluator.accuracy import AccuracyBase
from olive.evaluator.metric_config import LatencyMetricConfig, MetricGoal, get_user_config_class


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


# TODO: support multiple subtypes at the same type for the same type
# Otherwise it's a waste of compute and time if we have to evaluate a model for different subtypes
# names, subtypes: Union[str, List[str]]
# However accuracy metric poses a slight problem since AUC has a different config. Need to resolve this
# so that we get a single metric config for a single type
# This way, the user can return multiple metrics at once
class Metric(ConfigBase):
    name: str
    type: MetricType
    sub_type: Union[AccuracySubType, LatencySubType] = None
    higher_is_better: bool = True
    priority_rank: int = 1
    goal: MetricGoal = None
    metric_config: ConfigBase = None
    user_config: ConfigBase = None
    data_config: DataConfig = DataConfig()

    @validator("sub_type", always=True, pre=True)
    def validate_sub_type(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        if values["type"] == MetricType.CUSTOM:
            return None
        sub_type_enum = AccuracySubType if values["type"] == MetricType.ACCURACY else LatencySubType
        try:
            v = sub_type_enum(v)
        except ValueError:
            raise ValueError(
                f"sub_type must be one of {list(sub_type_enum.__members__.keys())} for {values['type']} metric"
            )
        return v

    @validator("higher_is_better", always=True, pre=True)
    def validate_higher_is_better(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        if values["type"] == MetricType.ACCURACY:
            return True
        if values["type"] == MetricType.LATENCY:
            return False
        if v is None:
            raise ValueError("higher_is_better must be specified for custom metric")
        return v

    @validator("metric_config", always=True, pre=True)
    def validate_metric_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")
        if "sub_type" not in values:
            raise ValueError("Invalid sub_type")

        if values["type"] == MetricType.CUSTOM:
            return None

        # metric config class
        if values["type"] == MetricType.LATENCY:
            metric_config_class = LatencyMetricConfig
        elif values["type"] == MetricType.ACCURACY:
            metric_config_class = AccuracyBase.registry[values["sub_type"]].get_config_class()

        # validate metric config
        return validate_config(v, ConfigBase, metric_config_class)

    @validator("user_config", pre=True)
    def validate_user_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        user_config_class = get_user_config_class(values["type"])
        return validate_config(v, ConfigBase, user_config_class)

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


class MetricList(BaseModel):
    __root__: List[Metric]
