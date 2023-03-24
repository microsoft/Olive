# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_custom_metric, get_latency_metric
from unittest.mock import MagicMock, patch

import pytest

from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricType
from olive.systems.local import LocalSystem


class TestLocalSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.local_system = LocalSystem()

    def test_run_pass(self):
        # setup
        p = MagicMock()
        olive_model = MagicMock()
        output_model_path = "output_model_path"

        # execute
        self.local_system.run_pass(p, olive_model, output_model_path)

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
    @patch("olive.systems.local.evaluate_accuracy")
    @patch("olive.systems.local.evaluate_latency")
    @patch("olive.systems.local.evaluate_custom_metric")
    def test_evaluate_model(self, mock_evaluate_custom_metric, mock_evaluate_latency, mock_evaluate_accuracy, metric):
        # setup
        olive_model = MagicMock()
        expected_res = "0.382715310"
        mock_evaluate_custom_metric.return_value = expected_res
        mock_evaluate_latency.return_value = expected_res
        mock_evaluate_accuracy.return_value = expected_res

        # execute
        actual_res = self.local_system.evaluate_model(olive_model, [metric])

        # assert
        if metric.type == MetricType.ACCURACY:
            mock_evaluate_accuracy.called_once_with(olive_model, metric, self.local_system.device)
        if metric.type == MetricType.LATENCY:
            mock_evaluate_latency.called_once_with(olive_model, metric, self.local_system.device)
        if metric.type == MetricType.CUSTOM:
            mock_evaluate_custom_metric.called_once_with(olive_model, metric, self.local_system.device)
        assert actual_res[metric.name] == expected_res
