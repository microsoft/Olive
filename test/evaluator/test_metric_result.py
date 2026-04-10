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
        # setup

        # execute
        result = SubMetricResult(value=0.95, priority=1, higher_is_better=True)

        # assert
        assert result.value == 0.95
        assert result.priority == 1
        assert result.higher_is_better is True

    def test_integer_value(self):
        # setup

        # execute
        result = SubMetricResult(value=100, priority=2, higher_is_better=False)

        # assert
        assert result.value == 100

    def test_float_value(self):
        # setup

        # execute
        result = SubMetricResult(value=0.001, priority=0, higher_is_better=True)

        # assert
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
        # setup
        result = self._make_result()

        # execute
        accuracy = result.get_value("accuracy", "top1")
        latency = result.get_value("latency", "avg")

        # assert
        assert accuracy == 0.95
        assert latency == 10.5

    def test_get_all_sub_type_metric_value(self):
        # setup
        result = self._make_result()

        # execute
        latency_values = result.get_all_sub_type_metric_value("latency")

        # assert
        assert latency_values == {"avg": 10.5, "p99": 20.0}

    def test_get_all_sub_type_single_metric(self):
        # setup
        result = self._make_result()

        # execute
        accuracy_values = result.get_all_sub_type_metric_value("accuracy")

        # assert
        assert accuracy_values == {"top1": 0.95}

    def test_str_representation(self):
        # setup
        result = self._make_result()

        # execute
        result_str = str(result)
        parsed = json.loads(result_str)

        # assert
        assert parsed["accuracy-top1"] == 0.95
        assert parsed["latency-avg"] == 10.5

    def test_len(self):
        # setup
        result = self._make_result()

        # execute
        length = len(result)

        # assert
        assert length == 3

    def test_getitem(self):
        # setup
        result = self._make_result()

        # execute
        item = result["accuracy-top1"]

        # assert
        assert item.value == 0.95

    def test_delimiter(self):
        # setup

        # execute
        delimiter = MetricResult.delimiter

        # assert
        assert delimiter == "-"


class TestJointMetricKey:
    def test_basic(self):
        # setup

        # execute
        result = joint_metric_key("accuracy", "top1")

        # assert
        assert result == "accuracy-top1"

    def test_with_special_names(self):
        # setup

        # execute
        result = joint_metric_key("latency", "p99")

        # assert
        assert result == "latency-p99"


class TestFlattenMetricSubType:
    def test_flatten(self):
        # setup
        metric_dict = {
            "accuracy": {"top1": {"value": 0.95, "priority": 1, "higher_is_better": True}},
            "latency": {"avg": {"value": 10.5, "priority": 2, "higher_is_better": False}},
        }

        # execute
        result = flatten_metric_sub_type(metric_dict)

        # assert
        assert "accuracy-top1" in result
        assert "latency-avg" in result


class TestFlattenMetricResult:
    def test_flatten_to_metric_result(self):
        # setup
        dict_results = {
            "accuracy": {
                "top1": {"value": 0.95, "priority": 1, "higher_is_better": True},
            },
            "latency": {
                "avg": {"value": 10.5, "priority": 2, "higher_is_better": False},
            },
        }

        # execute
        result = flatten_metric_result(dict_results)

        # assert
        assert isinstance(result, MetricResult)
        assert result.get_value("accuracy", "top1") == 0.95
        assert result.get_value("latency", "avg") == 10.5
