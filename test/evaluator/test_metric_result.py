# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

from olive.evaluator.metric_result import (
    MetricResult,
    SubMetricResult,
    flatten_metric_result,
    flatten_metric_sub_type,
    joint_metric_key,
)


class TestSubMetricResult:
    def test_creation(self):
        result = SubMetricResult(value=0.95, priority=1, higher_is_better=True)
        assert result.value == 0.95
        assert result.priority == 1
        assert result.higher_is_better is True

    def test_integer_value(self):
        result = SubMetricResult(value=100, priority=2, higher_is_better=False)
        assert result.value == 100

    def test_float_value(self):
        result = SubMetricResult(value=0.001, priority=0, higher_is_better=True)
        assert result.value == 0.001


class TestMetricResult:
    def _make_result(self):
        return MetricResult.model_validate(
            {
                "accuracy-top1": SubMetricResult(value=0.95, priority=1, higher_is_better=True),
                "latency-avg": SubMetricResult(value=10.5, priority=2, higher_is_better=False),
                "latency-p99": SubMetricResult(value=20.0, priority=3, higher_is_better=False),
            }
        )

    def test_get_value(self):
        result = self._make_result()
        assert result.get_value("accuracy", "top1") == 0.95
        assert result.get_value("latency", "avg") == 10.5

    def test_get_all_sub_type_metric_value(self):
        result = self._make_result()
        latency_values = result.get_all_sub_type_metric_value("latency")
        assert latency_values == {"avg": 10.5, "p99": 20.0}

    def test_get_all_sub_type_single_metric(self):
        result = self._make_result()
        accuracy_values = result.get_all_sub_type_metric_value("accuracy")
        assert accuracy_values == {"top1": 0.95}

    def test_str_representation(self):
        result = self._make_result()
        result_str = str(result)
        parsed = json.loads(result_str)
        assert parsed["accuracy-top1"] == 0.95
        assert parsed["latency-avg"] == 10.5

    def test_len(self):
        result = self._make_result()
        assert len(result) == 3

    def test_getitem(self):
        result = self._make_result()
        assert result["accuracy-top1"].value == 0.95

    def test_delimiter(self):
        assert MetricResult.delimiter == "-"


class TestJointMetricKey:
    def test_basic(self):
        assert joint_metric_key("accuracy", "top1") == "accuracy-top1"

    def test_with_special_names(self):
        assert joint_metric_key("latency", "p99") == "latency-p99"


class TestFlattenMetricSubType:
    def test_flatten(self):
        metric_dict = {
            "accuracy": {"top1": {"value": 0.95, "priority": 1, "higher_is_better": True}},
            "latency": {"avg": {"value": 10.5, "priority": 2, "higher_is_better": False}},
        }
        result = flatten_metric_sub_type(metric_dict)
        assert "accuracy-top1" in result
        assert "latency-avg" in result


class TestFlattenMetricResult:
    def test_flatten_to_metric_result(self):
        dict_results = {
            "accuracy": {
                "top1": {"value": 0.95, "priority": 1, "higher_is_better": True},
            },
            "latency": {
                "avg": {"value": 10.5, "priority": 2, "higher_is_better": False},
            },
        }
        result = flatten_metric_result(dict_results)
        assert isinstance(result, MetricResult)
        assert result.get_value("accuracy", "top1") == 0.95
        assert result.get_value("latency", "avg") == 10.5
