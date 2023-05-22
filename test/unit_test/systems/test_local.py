# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_custom_metric, get_latency_metric
from unittest.mock import MagicMock, patch

import pytest

from olive.constants import Framework
from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricResult, MetricType, joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.local import LocalSystem


class TestLocalSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = LocalSystem()

    def test_run_pass(self):
        # setup
        p = MagicMock()
        olive_model = MagicMock()
        output_model_path = "output_model_path"

        # execute
        self.system.run_pass(p, olive_model, output_model_path)

        # assert
        p.run.called_once_with(olive_model, output_model_path, None)

    METRIC_TEST_CASE = [
        (get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)),
        (get_accuracy_metric(AccuracySubType.F1_SCORE)),
        (get_accuracy_metric(AccuracySubType.PRECISION)),
        (get_accuracy_metric(AccuracySubType.RECALL)),
        (get_accuracy_metric(AccuracySubType.AUC)),
        (get_latency_metric(LatencySubType.AVG)),
        (get_latency_metric(LatencySubType.MAX)),
        (get_latency_metric(LatencySubType.MIN)),
        (get_latency_metric(LatencySubType.P50)),
        (get_latency_metric(LatencySubType.P75)),
        (get_latency_metric(LatencySubType.P90)),
        (get_latency_metric(LatencySubType.P95)),
        (get_latency_metric(LatencySubType.P99)),
        (get_latency_metric(LatencySubType.P999)),
        (get_custom_metric()),
    ]

    @pytest.mark.parametrize(
        "metric",
        METRIC_TEST_CASE,
    )
    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.get_user_config")
    @patch("olive.evaluator.olive_evaluator.OnnxEvaluator._evaluate_accuracy")
    @patch("olive.evaluator.olive_evaluator.OnnxEvaluator._evaluate_latency")
    @patch("olive.evaluator.olive_evaluator.OnnxEvaluator._evaluate_custom")
    def test_evaluate_model(
        self, mock_evaluate_custom, mock_evaluate_latency, mock_evaluate_accuracy, mock_get_user_config, metric
    ):
        # setup
        olive_model = MagicMock()
        olive_model.framework = Framework.ONNX
        expected_res = MetricResult.parse_obj(
            {
                sub_metric.name: {
                    "value": 0.382715310,
                    "priority": sub_metric.priority,
                    "higher_is_better": sub_metric.higher_is_better,
                }
                for sub_metric in metric.sub_types
            }
        )
        mock_evaluate_custom.return_value = expected_res
        mock_evaluate_latency.return_value = expected_res
        mock_evaluate_accuracy.return_value = expected_res
        mock_get_user_config.return_value = (None, None, None)

        # execute
        actual_res = self.system.evaluate_model(olive_model, [metric])

        # assert
        if metric.type == MetricType.ACCURACY:
            mock_evaluate_accuracy.called_once_with(olive_model, metric, None, "cpu", "CPUExecutionProvider")
        if metric.type == MetricType.LATENCY:
            mock_evaluate_latency.called_once_with(olive_model, metric, None, "cpu", "CPUExecutionProvider")
        if metric.type == MetricType.CUSTOM:
            mock_evaluate_custom.called_once_with(olive_model, metric, None, None, "cpu", "CPUExecutionProvider")

        joint_keys = [joint_metric_key(metric.name, sub_metric.name) for sub_metric in metric.sub_types]
        for joint_key in joint_keys:
            assert actual_res[joint_key].value == 0.38271531
