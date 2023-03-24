# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model, get_pytorch_model
from unittest.mock import patch

import pytest

from olive.evaluator.evaluation import evaluate_accuracy, evaluate_latency
from olive.evaluator.metric import AccuracySubType, LatencySubType


class TestEvaluation:

    ACCURACY_TEST_CASE = [
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.evaluation.AccuracyScore",
            0.99,
        ),
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.evaluation.F1Score",
            0.99,
        ),
        (
            get_pytorch_model(),
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.evaluation.Precision",
            0.99,
        ),
        (get_pytorch_model(), get_accuracy_metric(AccuracySubType.RECALL), "olive.evaluator.evaluation.Recall", 0.99),
        (get_pytorch_model(), get_accuracy_metric(AccuracySubType.AUC), "olive.evaluator.evaluation.AUC", 0.99),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.evaluation.AccuracyScore",
            0.99,
        ),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.evaluation.F1Score",
            0.99,
        ),
        (
            get_onnx_model(),
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.evaluation.Precision",
            0.99,
        ),
        (get_onnx_model(), get_accuracy_metric(AccuracySubType.RECALL), "olive.evaluator.evaluation.Recall", 0.99),
        (get_onnx_model(), get_accuracy_metric(AccuracySubType.AUC), "olive.evaluator.evaluation.AUC", 0.99),
    ]

    @pytest.mark.parametrize(
        "olive_model,metric,acc_subtype,expected_res",
        ACCURACY_TEST_CASE,
    )
    def test_evaluate_accuracy(self, olive_model, metric, acc_subtype, expected_res):
        # setup
        with patch(acc_subtype) as mock_acc:
            mock_acc.return_value.evaluate.return_value = expected_res

            # execute
            actual_res = evaluate_accuracy(olive_model, metric)

            # assert
            mock_acc.return_value.evaluate.assert_called_once()
            assert expected_res == actual_res

    LATENCY_TEST_CASE = [
        (get_pytorch_model(), get_latency_metric(LatencySubType.AVG), 1),
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
        actual_res = evaluate_latency(olive_model, metric)

        # assert
        assert expected_res > actual_res
