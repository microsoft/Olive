# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import get_accuracy_metric, get_onnx_model_config, get_pytorch_model_config
from unittest.mock import patch

import pytest

from olive.evaluator.metric import SubMetricResult
from olive.evaluator.metric_backend import HuggingfaceMetrics
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.local import LocalSystem


class TestMetricBackend:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.preds = [["O", "O", "B-MISC", "I-MISC", "I-MISC", "I-MISC", "O"], ["B-PER", "I-PER", "O"]]
        self.targets = [["O", "O", "O", "B-MISC", "I-MISC", "I-MISC", "O"], ["B-PER", "I-PER", "O"]]
        self.compute_res = {
            "MISC": {"precision": 0.999, "recall": 1.0, "f1": 1.0, "number": 1},
            "PER": {"precision": 1.0, "recall": 1.0, "f1": 1.0, "number": 1},
            "overall_precision": 1.0,
            "overall_recall": 1.0,
            "overall_f1": 1.0,
            "overall_accuracy": 1.0,
        }

    @patch("evaluate.load")
    def test_huggingface_metric_config(self, loader):
        loader().compute.return_value = self.compute_res
        metric = get_accuracy_metric("seqeval", backend="huggingface_metrics")
        for idx, _ in enumerate(metric.sub_types):
            metric.sub_types[idx].metric_config.load_params = {"process_id": 0}
            metric.sub_types[idx].metric_config.compute_params = {"suffix": True}
            metric.sub_types[idx].metric_config.result_key = "MISC.precision"
        backend = HuggingfaceMetrics()
        actual_res = backend.measure(self.preds, self.targets, metric)
        for _, v in actual_res.items():
            assert v.value == 0.999

    HF_ACCURACY_TEST_CASE = [
        (
            get_pytorch_model_config(),
            get_accuracy_metric("accuracy", "f1", backend="huggingface_metrics"),
            0.99,
        ),
        (
            get_onnx_model_config(),
            get_accuracy_metric("accuracy", "f1", backend="huggingface_metrics"),
            0.99,
        ),
    ]

    @pytest.mark.parametrize(
        "model_config,metric,expected_res",
        HF_ACCURACY_TEST_CASE,
    )
    def test_evaluate_backend(self, model_config, metric, expected_res):
        from olive.evaluator.metric_backend import HuggingfaceMetrics

        with patch.object(HuggingfaceMetrics, "measure_sub_metric") as mock_measure:
            mock_measure.return_value = SubMetricResult(value=expected_res, higher_is_better=True, priority=-1)
            system = LocalSystem()

            # execute
            actual_res = system.evaluate_model(model_config, None, [metric], DEFAULT_CPU_ACCELERATOR)

            # assert
            mock_measure.call_count == len(metric.sub_types)
            for sub_type in metric.sub_types:
                assert expected_res == actual_res.get_value(metric.name, sub_type.name)
