# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model_config
from unittest.mock import MagicMock, patch

import pytest
import torch

from olive.constants import Framework
from olive.evaluator.metric import AccuracySubType, LatencySubType
from olive.evaluator.olive_evaluator import OliveEvaluator, OnnxEvaluator
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.ort_inference import ORTInferenceSystem
from olive.systems.ort_inference.ort_inference_system import ORTInferenceEvaluator

# pylint: disable=attribute-defined-outside-init, protected-access


def get_inference_system():
    # use the current python environment as the test environment
    executable_parent = Path(sys.executable).parent.resolve().as_posix()
    return ORTInferenceSystem(executable_parent)


class TestORTInferenceSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = get_inference_system()

    def test_get_supported_execution_providers(self):
        import onnxruntime as ort

        assert set(self.system.get_supported_execution_providers()) == set(ort.get_available_providers())

    def test_unsupported_pass_run(self):
        with pytest.raises(NotImplementedError):
            self.system.run_pass(None, None, None, None)

    @patch("olive.systems.ort_inference.ort_inference_system.ORTInferenceEvaluator.evaluate")
    def test_evaluate_model(self, mock_evaluate):
        olive_model_config = MagicMock()
        olive_model_config.type = "ONNXModel"
        olive_model = olive_model_config.create_model()
        olive_model.framework = Framework.ONNX

        metric = MagicMock()

        self.system.evaluate_model(olive_model_config, None, [metric], DEFAULT_CPU_ACCELERATOR)

        # assert
        mock_evaluate.assert_called_once_with(
            olive_model,
            None,
            [metric],
            device=DEFAULT_CPU_ACCELERATOR.accelerator_type,
            execution_providers=DEFAULT_CPU_ACCELERATOR.execution_provider,
        )

    def test_unsupported_evaluate_model(self):
        olive_model_config = MagicMock()
        olive_model_config.type = "PyTorchModel"

        with pytest.raises(ValueError) as errinfo:  # noqa: PT011
            self.system.evaluate_model(olive_model_config, None, None, None)

        assert "ORTInferenceSystem only supports evaluation for ONNXModel" in str(errinfo.value)


class TestORTInferenceEvaluator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = get_inference_system()
        self.evaluator = ORTInferenceEvaluator(self.system.environ)
        self.onnx_evaluator = OnnxEvaluator()

    def test__inference(self):
        model = get_onnx_model_config().create_model()
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, random_dataloader=False)
        metric = OliveEvaluator.generate_metric_user_config_with_model_io(metric, model)
        dataloader, _, post_func = OliveEvaluator.get_user_config(model.framework, None, metric)

        actual_model_output, actual_target = self.evaluator._inference(
            model,
            metric,
            dataloader,
            post_func=post_func,
            device=DEFAULT_CPU_ACCELERATOR.accelerator_type,
            execution_providers=DEFAULT_CPU_ACCELERATOR.execution_provider,
        )
        expected_model_output, expected_target = self.onnx_evaluator._inference(
            model,
            metric,
            dataloader,
            post_func=post_func,
            device=DEFAULT_CPU_ACCELERATOR.accelerator_type,
            execution_providers=DEFAULT_CPU_ACCELERATOR.execution_provider,
        )

        # ensure the results from the local run and the subprocess run are the same
        assert torch.equal(actual_model_output.preds, expected_model_output.preds)
        assert torch.equal(actual_model_output.logits, expected_model_output.logits)
        assert torch.equal(actual_target, expected_target)

    def test__evaluate_raw_latency(self):
        model = get_onnx_model_config().create_model()
        metric = get_latency_metric(LatencySubType.AVG)
        metric = OliveEvaluator.generate_metric_user_config_with_model_io(metric, model)
        dataloader, _, _ = OliveEvaluator.get_user_config(model.framework, None, metric)

        res = self.evaluator._evaluate_raw_latency(
            model,
            None,
            metric,
            dataloader,
            post_func=None,
            device=DEFAULT_CPU_ACCELERATOR.accelerator_type,
            execution_providers=DEFAULT_CPU_ACCELERATOR.execution_provider,
        )
        # ensure the results are positive floats
        assert all(val > 0 for val in res)

    # def test_inference_runner(self):
