# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import pytest
from pydantic import ValidationError

from olive.evaluator.metric_config import (
    LatencyMetricConfig,
    MetricGoal,
    SizeOnDiskMetricConfig,
    ThroughputMetricConfig,
    get_user_config_class,
)


class TestLatencyMetricConfig:
    def test_defaults(self):
        config = LatencyMetricConfig()
        assert config.warmup_num == 10
        assert config.repeat_test_num == 20
        assert config.sleep_num == 0

    def test_custom_values(self):
        config = LatencyMetricConfig(warmup_num=5, repeat_test_num=100, sleep_num=2)
        assert config.warmup_num == 5
        assert config.repeat_test_num == 100
        assert config.sleep_num == 2


class TestThroughputMetricConfig:
    def test_defaults(self):
        config = ThroughputMetricConfig()
        assert config.warmup_num == 10
        assert config.repeat_test_num == 20
        assert config.sleep_num == 0

    def test_custom_values(self):
        config = ThroughputMetricConfig(warmup_num=3, repeat_test_num=50, sleep_num=1)
        assert config.warmup_num == 3
        assert config.repeat_test_num == 50
        assert config.sleep_num == 1


class TestSizeOnDiskMetricConfig:
    def test_creation(self):
        config = SizeOnDiskMetricConfig()
        assert isinstance(config, SizeOnDiskMetricConfig)


class TestMetricGoal:
    def test_threshold_type(self):
        goal = MetricGoal(type="threshold", value=0.9)
        assert goal.type == "threshold"
        assert goal.value == 0.9

    def test_min_improvement_type(self):
        goal = MetricGoal(type="min-improvement", value=0.05)
        assert goal.type == "min-improvement"
        assert goal.value == 0.05

    def test_max_degradation_type(self):
        goal = MetricGoal(type="max-degradation", value=0.1)
        assert goal.type == "max-degradation"
        assert goal.value == 0.1

    def test_percent_min_improvement_type(self):
        goal = MetricGoal(type="percent-min-improvement", value=5.0)
        assert goal.type == "percent-min-improvement"

    def test_percent_max_degradation_type(self):
        goal = MetricGoal(type="percent-max-degradation", value=10.0)
        assert goal.type == "percent-max-degradation"

    def test_invalid_type_raises(self):
        with pytest.raises(ValidationError, match="Metric goal type must be one of"):
            MetricGoal(type="invalid_type", value=0.5)

    def test_negative_value_for_min_improvement_raises(self):
        with pytest.raises(ValidationError, match="Value must be nonnegative"):
            MetricGoal(type="min-improvement", value=-0.5)

    def test_negative_value_for_max_degradation_raises(self):
        with pytest.raises(ValidationError, match="Value must be nonnegative"):
            MetricGoal(type="max-degradation", value=-0.1)

    def test_negative_value_for_percent_min_improvement_raises(self):
        with pytest.raises(ValidationError, match="Value must be nonnegative"):
            MetricGoal(type="percent-min-improvement", value=-5.0)

    def test_negative_value_for_percent_max_degradation_raises(self):
        with pytest.raises(ValidationError, match="Value must be nonnegative"):
            MetricGoal(type="percent-max-degradation", value=-10.0)

    def test_threshold_allows_negative_value(self):
        goal = MetricGoal(type="threshold", value=-1.0)
        assert goal.value == -1.0

    def test_has_regression_goal_min_improvement(self):
        goal = MetricGoal(type="min-improvement", value=0.05)
        assert goal.has_regression_goal() is False

    def test_has_regression_goal_percent_min_improvement(self):
        goal = MetricGoal(type="percent-min-improvement", value=5.0)
        assert goal.has_regression_goal() is False

    def test_has_regression_goal_max_degradation_positive(self):
        goal = MetricGoal(type="max-degradation", value=0.1)
        assert goal.has_regression_goal() is True

    def test_has_regression_goal_max_degradation_zero(self):
        goal = MetricGoal(type="max-degradation", value=0.0)
        assert goal.has_regression_goal() is False

    def test_has_regression_goal_percent_max_degradation_positive(self):
        goal = MetricGoal(type="percent-max-degradation", value=10.0)
        assert goal.has_regression_goal() is True

    def test_has_regression_goal_threshold(self):
        goal = MetricGoal(type="threshold", value=0.9)
        assert goal.has_regression_goal() is False


class TestGetUserConfigClass:
    def test_custom_metric_type(self):
        cls = get_user_config_class("custom")
        instance = cls()
        assert hasattr(instance, "user_script")
        assert hasattr(instance, "evaluate_func")

    def test_unknown_metric_type(self):
        cls = get_user_config_class("latency")
        instance = cls()
        assert hasattr(instance, "user_script")
        # Unknown metric types still get common config
        assert hasattr(instance, "inference_settings")
