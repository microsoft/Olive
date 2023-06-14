# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Callable, Dict, List, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, ConfigParam, create_config_class
from olive.resource_path import OLIVE_RESOURCE_ANNOTATIONS, create_resource_path, get_local_path

WARMUP_NUM = 10
REPEAT_TEST_NUM = 20
SLEEP_NUM = 0

user_path_config = ["script_dir", "data_dir", "user_script"]
_common_user_config = {
    "script_dir": ConfigParam(type_=OLIVE_RESOURCE_ANNOTATIONS, is_path=True),
    "user_script": ConfigParam(type_=OLIVE_RESOURCE_ANNOTATIONS, is_path=True),
    "data_dir": ConfigParam(type_=OLIVE_RESOURCE_ANNOTATIONS, is_path=True),
    "batch_size": ConfigParam(type_=int, default_value=1),
    "input_names": ConfigParam(type_=List),
    "input_shapes": ConfigParam(type_=List),
    "input_types": ConfigParam(type_=List),
}

_common_user_config_validators = {}

_type_to_user_config = {
    "latency": {
        "dataloader_func": ConfigParam(type_=Union[Callable, str], is_object=True),
        "inference_settings": ConfigParam(type_=dict),
        "io_bind": ConfigParam(type_=bool, default_value=False),
    },
    "accuracy": {
        "dataloader_func": ConfigParam(type_=Union[Callable, str], is_object=True),
        "post_processing_func": ConfigParam(type_=Union[Callable, str], is_object=True),
        "inference_settings": ConfigParam(type_=dict),
    },
    "custom": {
        "evaluate_func": ConfigParam(type_=Union[Callable, str], required=True, is_object=True),
    },
}

_type_to_user_config_validators = {}


def get_user_config_class(metric_type: str):
    default_config = _common_user_config.copy()
    default_config.update(_type_to_user_config[metric_type])
    validators = _common_user_config_validators.copy()
    validators.update(_type_to_user_config_validators.get(metric_type, {}))
    return create_config_class(f"{metric_type.title()}UserConfig", default_config, ConfigBase, validators)


def get_properties_from_metric_type(metric_type):
    user_config_class = get_user_config_class(metric_type)
    # avoid to use schema() to get the fields, because it will skip the ones with object type
    return list(user_config_class.__fields__)


def localize_user_config(user_config: ConfigBase, localized_config: Dict = None) -> ConfigBase:
    user_config_dict = user_config.dict()
    localized_config = localized_config or {}
    for config_name in user_config_dict:
        default_config_setting = _common_user_config.get(config_name)
        if default_config_setting and default_config_setting.is_path:
            obj_config_value = getattr(user_config, config_name, None)
            if not obj_config_value:
                continue

            original_path = obj_config_value.get_path()
            local_path = localized_config.get(original_path)
            if not local_path:
                local_path = get_local_path(obj_config_value)
                localized_config[original_path] = local_path
            setattr(user_config, config_name, create_resource_path(local_path))
    return user_config, localized_config


# TODO: automate latency metric config also we standardize accuracy metric config
class LatencyMetricConfig(ConfigBase):
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
        if values["type"] in ["min-improvement", "max-degradation"] and v < 0:
            raise ValueError(f"Value must be positive for type {values['type']}")
        return v
