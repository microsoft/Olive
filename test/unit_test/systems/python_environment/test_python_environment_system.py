# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model
from unittest.mock import patch

import pytest

from olive.evaluator.evaluation import evaluate_accuracy
from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricType
from olive.evaluator.metric_config import MetricResult
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
        mock_evaluate_accuracy.return_value = MetricResult(
            key_for_rank=AccuracySubType.ACCURACY_SCORE,
            value_for_rank=0.9,
            metrics={AccuracySubType.ACCURACY_SCORE: 0.9},
        )
        mock_evaluate_latency.return_value = MetricResult(
            key_for_rank=LatencySubType.AVG,
            value_for_rank=10,
            metrics={LatencySubType.AVG: 10, LatencySubType.MAX: 20},
        )

        # execute
        res = self.system.evaluate_model(model, metrics)

        # assert
        assert res.signal["accuracy"].value_for_rank == 0.9
        assert res.signal["latency"].value_for_rank == 10
        assert mock_evaluate_accuracy.call_once_with(model, metrics[0])
        assert mock_evaluate_latency.call_once_with(model, metrics[1])

    @patch("olive.evaluator.evaluation.compute_accuracy")
    @patch("olive.systems.python_environment.python_environment_system.compute_accuracy")
    def test_evaluate_accuracy(self, mock_compute_accuracy, mock_compute_accuracy2):
        # setup
        model = get_onnx_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, False)
        mock_value = MetricResult(
            key_for_rank=metric.name,
            value_for_rank=0.9,
            metrics={metric.name: 0.9},
        )
        mock_compute_accuracy.return_value = mock_value
        mock_compute_accuracy2.return_value = mock_value

        # expected result
        expected_res = evaluate_accuracy(model, metric)

        # execute
        actual_res = self.system.evaluate_accuracy(model, metric)

        # assert
        assert actual_res.value_for_rank == expected_res.value_for_rank
        assert mock_compute_accuracy.call_args.args[0] == mock_compute_accuracy2.call_args.args[0]

    @patch("olive.systems.python_environment.python_environment_system.compute_latency")
    def test_evaluate_latency(self, mock_compute_latency):
        # setup
        model = get_onnx_model()
        metric = get_latency_metric(LatencySubType.AVG)
        metric_config = metric.metric_config[MetricType.LATENCY]
        metric_config.repeat_test_num = 5
        mock_compute_latency.return_value = MetricResult(
            key_for_rank="avg",
            value_for_rank=10,
            metrics={"avg": 10},
        )
        # expected result
        expected_res = 10

        # execute
        actual_res = self.system.evaluate_latency(model, metric)

        # assert
        assert actual_res.value_for_rank == expected_res
        assert len(mock_compute_latency.call_args.args[0]) == metric_config.repeat_test_num
        assert all([latency > 0 for latency in mock_compute_latency.call_args.args[0]])
