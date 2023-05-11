# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model
from unittest.mock import patch

import pytest

from olive.evaluator.metric import AccuracySubType, LatencySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorFactory
from olive.hardware import DEFAULT_CPU_ACCELERATOR
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
        mock_evaluate_accuracy.return_value = 0.9
        mock_evaluate_latency.return_value = 10

        # execute
        res = self.system.evaluate_model(model, metrics, DEFAULT_CPU_ACCELERATOR)

        # assert
        assert res == {"accuracy": 0.9, "latency": 10}
        assert mock_evaluate_accuracy.call_once_with(model, metrics[0])
        assert mock_evaluate_latency.call_once_with(model, metrics[1])

    @pytest.mark.skip(reason="Unable to patch static function calls from another function")
    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy")
    @patch("olive.systems.python_environment.python_environment_system.OliveEvaluator.compute_accuracy")
    def test_evaluate_accuracy(self, mock_compute_accuracy1, mock_compute_accuracy2):
        # setup
        model = get_onnx_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, False)
        mock_compute_accuracy1.return_value = 0.9
        mock_compute_accuracy2.return_value = 0.9

        # expected result
        evaluator = OliveEvaluatorFactory.create_evaluator_for_model(model)
        expected_res = evaluator.evaluate(model, [metric])[metric.name]

        # execute
        actual_res = self.system.evaluate_accuracy(model, metric)

        # assert
        assert actual_res == expected_res
        assert mock_compute_accuracy1.call_args.args[1] == mock_compute_accuracy2.call_args.args[1]

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_latency")
    def test_evaluate_latency(self, mock_compute_latency):
        # setup
        model = get_onnx_model()
        metric = get_latency_metric(LatencySubType.AVG)
        metric.metric_config.repeat_test_num = 5
        mock_compute_latency.return_value = 10

        # expected result
        expected_res = 10

        # execute
        actual_res = self.system.evaluate_latency(model, metric)

        # assert
        assert actual_res == expected_res
        assert len(mock_compute_latency.call_args.args[1]) == metric.metric_config.repeat_test_num
        assert all([latency > 0 for latency in mock_compute_latency.call_args.args[1]])
