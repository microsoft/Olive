# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model, get_pytorch_model
from unittest.mock import patch

import pytest

from olive.evaluator.metric import AccuracySubType, LatencySubType
from olive.systems.local import LocalSystem


class TestOliveEvaluator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = LocalSystem()

    ACCURACY_TEST_CASE = [
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (get_pytorch_model(), get_accuracy_metric(AccuracySubType.RECALL), "olive.evaluator.accuracy.Recall", 0.99),
        (get_pytorch_model(), get_accuracy_metric(AccuracySubType.AUC), "olive.evaluator.accuracy.AUC", 0.99),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (get_onnx_model(), get_accuracy_metric(AccuracySubType.RECALL), "olive.evaluator.accuracy.Recall", 0.99),
        (get_onnx_model(), get_accuracy_metric(AccuracySubType.AUC), "olive.evaluator.accuracy.AUC", 0.99),
    ]

    @pytest.mark.parametrize(
        "olive_model,metric,acc_subtype,expected_res",
        ACCURACY_TEST_CASE,
    )
    def test_evaluate_accuracy(self, olive_model, metric, acc_subtype, expected_res):
        # setup
        with patch(f"{acc_subtype}.measure") as mock_acc:
            mock_acc.return_value = expected_res

            # execute
            actual_res = self.system.evaluate_model(olive_model, [metric])[metric.name]

            # assert
            mock_acc.assert_called_once()

            for sub_type in metric.sub_types:
                assert expected_res == actual_res[sub_type.name].value

    LATENCY_TEST_CASE = [
        (get_pytorch_model(), get_latency_metric(LatencySubType.AVG, LatencySubType.MAX), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.MAX), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.MIN), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P50), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P75), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P90), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P95), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P99), 1),
        (get_pytorch_model(), get_latency_metric(LatencySubType.P999), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.AVG), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.MAX), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.MIN), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P50), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P75), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P90), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P95), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P99), 1),
        (get_onnx_model(), get_latency_metric(LatencySubType.P999), 1),
    ]

    @pytest.mark.parametrize(
        "olive_model,metric,expected_res",
        LATENCY_TEST_CASE,
    )
    def test_evaluate_latency(self, olive_model, metric, expected_res):
        # execute
        actual_res = self.system.evaluate_model(olive_model, [metric])[metric.name]

        for sub_type in metric.sub_types:
            assert expected_res > actual_res[sub_type.name].value
