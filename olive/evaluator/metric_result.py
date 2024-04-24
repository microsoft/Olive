# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from typing import ClassVar, Dict, Union

from olive.common.config_utils import ConfigBase, ConfigDictBase


class SubMetricResult(ConfigBase):
    value: Union[float, int]
    priority: int
    higher_is_better: bool


class MetricResult(ConfigDictBase):
    __root__: Dict[str, SubMetricResult]
    delimiter: ClassVar[str] = "-"

    def get_value(self, metric_name, sub_type_name):
        if not self.__root__:
            return None
        return self.__root__[joint_metric_key(metric_name, sub_type_name)].value

    def get_all_sub_type_metric_value(self, metric_name):
        return {k.split(self.delimiter)[-1]: v.value for k, v in self.__root__.items() if k.startswith(metric_name)}

    def __str__(self) -> str:
        repr_obj = {k: v.value for k, v in self.__root__.items()}
        return json.dumps(repr_obj, indent=2)


def joint_metric_key(metric_name, sub_type_name):
    return f"{metric_name}{MetricResult.delimiter}{sub_type_name}"


def flatten_metric_sub_type(metric_dict: Dict[str, Dict]):
    flatten_results = {}
    for metric_name, metric_res in metric_dict.items():
        for sub_type_name, sub_type_res in metric_res.items():
            key = joint_metric_key(metric_name, sub_type_name)
            flatten_results[key] = sub_type_res
    return flatten_results


def flatten_metric_result(dict_results: Dict[str, MetricResult]):
    return MetricResult.parse_obj(flatten_metric_sub_type(dict_results))
