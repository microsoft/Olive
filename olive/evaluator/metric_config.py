# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Callable, ClassVar, Dict, List, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, ConfigDictBase, ConfigParam, create_config_class

WARMUP_NUM = 10
REPEAT_TEST_NUM = 20
SLEEP_NUM = 0


_common_user_config = {
    "script_dir": ConfigParam(type_=Union[Path, str]),
    "user_script": ConfigParam(type_=Union[Path, str]),
    "data_dir": ConfigParam(type_=Union[Path, str]),
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


class SubTypeMetricResult(ConfigBase):
    value: Union[float, int]
    priority_rank: int
    higher_is_better: bool


class MetricResult(ConfigDictBase):
    __root__: Dict[str, SubTypeMetricResult] = None
    delimiter: ClassVar[str] = "-"

    def get_value(self, metric_name, sub_type_name):
        if not self.__root__:
            return None
        return self.__root__[joint_metric_key(metric_name, sub_type_name)].value

    def get_all_sub_type_metric_value(self, metric_name):
        return {k.split(self.delimiter)[-1]: v.value for k, v in self.__root__.items() if k.startswith(metric_name)}

    def __str__(self) -> str:
        repr_obj = {k: v.value for k, v in self.__root__.items()}
        return f"{repr_obj}"


def joint_metric_key(metric_name, sub_type_name):
    return f"{metric_name}{MetricResult.delimiter}{sub_type_name}"


def flatten_metric_sub_type(metric_dict: Dict[str, Dict]):
    flatten_results = {}
    for metric_name, metric_res in metric_dict.items():
        for sub_type_name, sub_type_res in metric_res.items():
            key = f"{metric_name}{MetricResult.delimiter}{sub_type_name}"
            flatten_results[key] = sub_type_res
    return flatten_results


def flatten_metric_result(dict_results: Dict[str, MetricResult]):
    return MetricResult.parse_obj(flatten_metric_sub_type(dict_results))
