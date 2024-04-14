# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Union

from olive.common.config_utils import ConfigBase, ConfigParam, ParamCategory, create_config_class
from olive.common.pydantic_v1 import validator
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS

logger = logging.getLogger(__name__)

WARMUP_NUM = 10
REPEAT_TEST_NUM = 20
SLEEP_NUM = 0

_common_user_config = {
    "script_dir": ConfigParam(type_=Union[Path, str]),
    "user_script": ConfigParam(type_=Union[Path, str]),
    "inference_settings": ConfigParam(type_=dict),
    "data_dir": ConfigParam(type_=OLIVE_RESOURCE_ANNOTATIONS, category=ParamCategory.DATA),
    "dataloader_func": ConfigParam(type_=Union[Callable, str], category=ParamCategory.OBJECT),
    "func_kwargs": ConfigParam(type_=Dict[str, Dict[str, Any]]),
    "batch_size": ConfigParam(type_=int, default_value=1),
    "input_names": ConfigParam(type_=List),
    "input_shapes": ConfigParam(type_=List),
    "input_types": ConfigParam(type_=List),
    "shared_kv_buffer": ConfigParam(type_=bool, default_value=False),
    "io_bind": ConfigParam(type_=bool, default_value=False),
}

_common_user_config_validators = {}

_type_to_user_config = {
    "accuracy": {
        "post_processing_func": ConfigParam(type_=Union[Callable, str], category=ParamCategory.OBJECT),
    },
    "custom": {
        "post_processing_func": ConfigParam(type_=Union[Callable, str], category=ParamCategory.OBJECT),
        "evaluate_func": ConfigParam(type_=Union[Callable, str], required=False, category=ParamCategory.OBJECT),
        "metric_func": ConfigParam(type_=Union[Callable, str], required=False, category=ParamCategory.OBJECT),
    },
}

_type_to_user_config_validators = {}


def get_user_config_class(metric_type: str):
    default_config = _common_user_config.copy()
    default_config.update(_type_to_user_config.get(metric_type, {}))
    validators = _common_user_config_validators.copy()
    validators.update(_type_to_user_config_validators.get(metric_type, {}))
    return create_config_class(f"{metric_type.title()}UserConfig", default_config, ConfigBase, validators)


def get_user_config_properties_from_metric_type(metric_type):
    user_config_class = get_user_config_class(metric_type)
    # avoid to use schema() to get the fields, because it will skip the ones with object type
    return list(user_config_class.__fields__)


# TODO(jambayk): automate latency metric config also we standardize accuracy metric config
class LatencyMetricConfig(ConfigBase):
    warmup_num: int = WARMUP_NUM
    repeat_test_num: int = REPEAT_TEST_NUM
    sleep_num: int = SLEEP_NUM


class ThroughputMetricConfig(ConfigBase):
    warmup_num: int = WARMUP_NUM
    repeat_test_num: int = REPEAT_TEST_NUM
    sleep_num: int = SLEEP_NUM


class MetricGoal(ConfigBase):
    type: str  # threshold , deviation, percent-deviation
    value: float

    @validator("type")
    def check_type(cls, v):
        allowed_types = [
            "threshold",
            "min-improvement",
            "percent-min-improvement",
            "max-degradation",
            "percent-max-degradation",
        ]
        if v not in allowed_types:
            raise ValueError(f"Metric goal type must be one of {allowed_types}")
        return v

    @validator("value")
    def check_value(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")
        if (
            values["type"]
            in {"min-improvement", "max-degradation", "percent-min-improvement", "percent-max-degradation"}
            and v < 0
        ):
            raise ValueError(f"Value must be nonnegative for type {values['type']}")
        return v

    def has_regression_goal(self):
        if self.type in {"min-improvement", "percent-min-improvement"}:
            return False
        elif self.type in {"max-degradation", "percent-max-degradation"}:
            return self.value > 0

        if self.type == "threshold":
            logger.warning("Metric goal type is threshold, Olive cannot determine if it is a regression goal or not.")
            return False
        return None
