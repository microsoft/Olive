# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_custom_metric, get_latency_metric
from unittest.mock import MagicMock, patch

import pytest

from olive.constants import Framework
from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricType
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
        expected_res = "0.382715310"
        mock_evaluate_custom.return_value = expected_res
        mock_evaluate_latency.return_value = expected_res
        mock_evaluate_accuracy.return_value = expected_res
        mock_get_user_config.return_value = (None, None, None)

        # execute
        actual_res = self.system.evaluate_model(olive_model, [metric])[metric.name]

        # assert
        if metric.type == MetricType.ACCURACY:
            mock_evaluate_accuracy.called_once_with(olive_model, metric, None, self.system.device, None)
        elif metric.type == MetricType.LATENCY:
            mock_evaluate_latency.called_once_with(olive_model, metric, None, self.system.device, None)
        elif metric.type == MetricType.CUSTOM:
            mock_evaluate_custom.called_once_with(olive_model, metric, None, None, self.system.device, None)

        assert actual_res == expected_res
