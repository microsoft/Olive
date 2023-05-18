# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model
from unittest.mock import patch

import pytest

from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricResult, MetricType, joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorFactory
from olive.systems.python_environment import PythonEnvironmentSystem


class TestPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the current python environment as the test environment
        executable_parent = Path(sys.executable).parent.resolve().as_posix()
        self.system = PythonEnvironmentSystem(executable_parent)

    def test_available_eps(self):
        import onnxruntime as ort

        assert set(self.system.get_available_eps()) == set(ort.get_available_providers())

    @patch("olive.systems.python_environment.PythonEnvironmentSystem.evaluate_accuracy")
    @patch("olive.systems.python_environment.PythonEnvironmentSystem.evaluate_latency")
    def test_evaluate_model(self, mock_evaluate_latency, mock_evaluate_accuracy):
        # setup
        model = get_onnx_model()
        metrics = [get_accuracy_metric(AccuracySubType.ACCURACY_SCORE), get_latency_metric(LatencySubType.AVG)]

        metrics_key = [
            joint_metric_key(metric.name, sub_metric.name) for metric in metrics for sub_metric in metric.sub_types
        ]

        mock_return_value = {
            sub_metric.name: {
                "value": 0.9 if metric.type == MetricType.ACCURACY else 10,
                "priority": sub_metric.priority,
                "higher_is_better": sub_metric.higher_is_better,
            }
            for metric in metrics
            for sub_metric in metric.sub_types
        }

        mock_evaluate_accuracy.return_value = MetricResult.parse_obj(
            {AccuracySubType.ACCURACY_SCORE: mock_return_value[AccuracySubType.ACCURACY_SCORE]}
        )
        mock_evaluate_latency.return_value = MetricResult.parse_obj(
            {LatencySubType.AVG: mock_return_value[LatencySubType.AVG]}
        )

        # execute
        res = self.system.evaluate_model(model, metrics)

        # assert
        assert res[metrics_key[0]].value == 0.9
        assert res[metrics_key[1]].value == 10
        assert mock_evaluate_accuracy.call_once_with(model, metrics[0])
        assert mock_evaluate_latency.call_once_with(model, metrics[1])

    @pytest.mark.skip(reason="Unable to patch static function calls from another function")
    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy")
    @patch("olive.systems.python_environment.python_environment_system.OliveEvaluator.compute_accuracy")
    def test_evaluate_accuracy(self, mock_compute_accuracy1, mock_compute_accuracy2):
        # setup
        model = get_onnx_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, random_dataloader=False)
        mock_value = MetricResult.parse_obj(
            {
                AccuracySubType.ACCURACY_SCORE: {
                    "value": 0.9,
                    "priority": 1,
                    "higher_is_better": True,
                }
            }
        )

        mock_compute_accuracy1.return_value = mock_value
        mock_compute_accuracy2.return_value = mock_value

        # expected result
        evaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        expected_res = evaluator.evaluate(model, [metric])[metric.name]

        # execute
        actual_res = self.system.evaluate_accuracy(model, metric)

        # assert
        assert actual_res[AccuracySubType.ACCURACY_SCORE].value == expected_res[AccuracySubType.ACCURACY_SCORE].value
        assert mock_compute_accuracy1.call_args.args[1] == mock_compute_accuracy2.call_args.args[1]

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_latency")
    def test_evaluate_latency(self, mock_compute_latency):
        # setup
        model = get_onnx_model()
        metric = get_latency_metric(LatencySubType.AVG)
        metric_config = metric.sub_types[0].metric_config
        metric_config.repeat_test_num = 5

        mock_value = MetricResult.parse_obj(
            {
                LatencySubType.AVG: {
                    "value": 10,
                    "priority": 1,
                    "higher_is_better": True,
                }
            }
        )

        mock_compute_latency.return_value = mock_value
        # expected result
        expected_res = 10

        # execute
        actual_res = self.system.evaluate_latency(model, metric)

        # assert
        assert actual_res[LatencySubType.AVG].value == expected_res
        assert len(mock_compute_latency.call_args.args[1]) == metric_config.repeat_test_num
        assert all([latency > 0 for latency in mock_compute_latency.call_args.args[1]])
