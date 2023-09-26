# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pickle
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model, get_onnx_model_config
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricResult, MetricType, joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.local import LocalSystem
from olive.systems.python_environment import PythonEnvironmentSystem
from olive.systems.python_environment.available_eps import main as available_eps_main
from olive.systems.python_environment.inference_runner import main as inference_runner_main
from olive.systems.python_environment.is_valid_ep import main as is_valid_ep_main


class TestPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the current python environment as the test environment
        executable_parent = Path(sys.executable).parent.resolve().as_posix()
        self.system = PythonEnvironmentSystem(executable_parent)

    def test_get_supported_execution_providers(self):
        import onnxruntime as ort

        assert set(self.system.get_supported_execution_providers()) == set(ort.get_available_providers())

    @patch("olive.systems.python_environment.PythonEnvironmentSystem.evaluate_accuracy")
    @patch("olive.systems.python_environment.PythonEnvironmentSystem.evaluate_latency")
    # mock generate_metric_user_config_with_model_io to return the input metric
    @patch(
        "olive.evaluator.olive_evaluator.OliveEvaluator.generate_metric_user_config_with_model_io",
        side_effect=lambda x, _: x,
    )
    def test_evaluate_model(self, _, mock_evaluate_latency, mock_evaluate_accuracy):
        # setup
        model = get_onnx_model()
        model_config = MagicMock()
        model_config.type = "ONNXModel"
        model_config.create_model.return_value = model
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
        res = self.system.evaluate_model(model_config, None, metrics, DEFAULT_CPU_ACCELERATOR)

        # assert
        assert res[metrics_key[0]].value == 0.9
        assert res[metrics_key[1]].value == 10
        mock_evaluate_accuracy.assert_called_once_with(model, None, metrics[0], DEFAULT_CPU_ACCELERATOR)
        mock_evaluate_latency.assert_called_once_with(model, None, metrics[1], DEFAULT_CPU_ACCELERATOR)

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy")
    def test_evaluate_accuracy(self, mock_compute_accuracy):
        # setup
        model_config = get_onnx_model_config()
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
        mock_compute_accuracy.return_value = mock_value

        # expected result
        local_system = LocalSystem()
        expected_res = local_system.evaluate_model(model_config, None, [metric], DEFAULT_CPU_ACCELERATOR)[
            "accuracy-accuracy_score"
        ]

        # execute
        actual_res = self.system.evaluate_model(model_config, None, [metric], DEFAULT_CPU_ACCELERATOR)[
            "accuracy-accuracy_score"
        ]

        # assert
        assert actual_res == expected_res
        assert mock_compute_accuracy.call_count == 2
        # local system call
        expected_call = mock_compute_accuracy.mock_calls[0]
        # python environment call
        actual_call = mock_compute_accuracy.mock_calls[1]
        assert actual_call.args[0] == expected_call.args[0]
        assert torch.equal(actual_call.args[1].preds, expected_call.args[1].preds)
        assert torch.equal(actual_call.args[1].logits, expected_call.args[1].logits)
        assert torch.equal(actual_call.args[2], expected_call.args[2])

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_latency")
    def test_evaluate_latency(self, mock_compute_latency):
        # setup
        model_config = get_onnx_model_config()
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
        actual_res = self.system.evaluate_latency(model_config.create_model(), None, metric, DEFAULT_CPU_ACCELERATOR)

        # assert
        assert actual_res[LatencySubType.AVG].value == expected_res
        assert len(mock_compute_latency.call_args.args[1]) == metric_config.repeat_test_num
        assert all(latency > 0 for latency in mock_compute_latency.call_args.args[1])

    @patch("onnxruntime.get_available_providers")
    def test_available_eps_script(self, mock_get_providers, tmp_path):
        mock_get_providers.return_value = ["CPUExecutionProvider"]
        output_path = tmp_path / "available_eps.pkl"

        # command
        args = ["--output_path", str(output_path)]

        # execute
        available_eps_main(args)

        # assert
        assert output_path.exists()
        mock_get_providers.assert_called_once()
        with output_path.open("rb") as f:
            assert pickle.load(f) == ["CPUExecutionProvider"]

    @pytest.mark.parametrize("valid", [True, False])
    @patch("olive.systems.python_environment.is_valid_ep.get_ort_inference_session")
    def test_is_valid_ep_script(self, mock_get_session, tmp_path, valid):
        if valid:
            mock_get_session.return_value = None
        else:
            mock_get_session.side_effect = Exception("Mock Failure")
        output_path = tmp_path / "is_valid_ep.pkl"

        # command
        args = ["--model_path", "model.onnx", "--ep", "CPUExecutionProvider", "--output_path", str(output_path)]

        # execute
        is_valid_ep_main(args)

        # assert
        assert output_path.exists()
        mock_get_session.assert_called_once_with("model.onnx", {"execution_provider": "CPUExecutionProvider"})
        with output_path.open("rb") as f:
            if valid:
                assert pickle.load(f) == {"valid": True}
            else:
                assert pickle.load(f) == {"valid": False, "error": "Mock Failure"}

    @patch("olive.systems.python_environment.inference_runner.get_ort_inference_session")
    def test_inference_runner_script_accuracy(self, mock_get_session, tmp_path):
        # setup
        mock_inference_session = MagicMock()
        dummy_output = np.array([1, 2, 3])
        mock_inference_session.run.return_value = dummy_output
        mock_get_session.return_value = mock_inference_session

        model = "model.onnx"
        inference_settings = {"execution_provider": "CPUExecutionProvider"}
        inference_settings_path = tmp_path / "inference_settings.pkl"
        with inference_settings_path.open("wb") as f:
            pickle.dump(inference_settings, f)
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        num_batches = 2
        for i in range(num_batches):
            np.savez(input_dir / f"input_{i}.npz", input=np.array([i]))
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # command
        args = [
            "--type",
            "accuracy",
            "--model_path",
            model,
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--inference_settings_path",
            str(inference_settings_path),
            "--num_batches",
            str(num_batches),
        ]

        # execute
        inference_runner_main(args)

        # assert
        mock_get_session.assert_called_once_with(model, inference_settings)
        assert mock_inference_session.run.call_count == num_batches
        assert mock_inference_session.run.mock_calls == [
            mock.call(input_feed={"input": np.array([i])}, output_names=None) for i in range(num_batches)
        ]
        for i in range(num_batches):
            assert (output_dir / f"output_{i}.npy").exists()
            assert np.load(output_dir / f"output_{i}.npy") == dummy_output

    @pytest.mark.parametrize("io_bind", [True, False])
    @patch("olive.systems.python_environment.inference_runner.get_ort_inference_session")
    def test_inference_runner_script_latency(self, mock_get_session, tmp_path, io_bind):
        # setup
        # inference session
        mock_inference_session = MagicMock()
        mock_inference_session.run.return_value = None
        mock_inference_session.run_with_iobinding.return_value = None
        # io binding
        io_bind_op = MagicMock()
        mock_inference_session.io_binding.return_value = io_bind_op
        # get session
        mock_get_session.return_value = mock_inference_session

        model = "model.onnx"
        inference_settings = {"execution_provider": "CPUExecutionProvider"}
        inference_settings_path = tmp_path / "inference_settings.pkl"
        with inference_settings_path.open("wb") as f:
            pickle.dump(inference_settings, f)
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        np.savez(input_dir / "input.npz", input=np.array([1]))
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)
        warmup_num = 1
        repeat_test_num = 3

        # command
        args = [
            "--type",
            "latency",
            "--model_path",
            model,
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
            "--inference_settings_path",
            str(inference_settings_path),
            "--warmup_num",
            str(warmup_num),
            "--repeat_test_num",
            str(repeat_test_num),
        ]
        if io_bind:
            args += ["--io_bind", "--device", "cpu"]

        # execute
        inference_runner_main(args)

        # assert
        total_num = warmup_num + repeat_test_num
        mock_get_session.assert_called_once_with(model, inference_settings)
        if not io_bind:
            assert mock_inference_session.run.call_count == total_num
            assert mock_inference_session.run.mock_calls == [
                mock.call(input_feed={"input": np.array([1])}, output_names=None) for _ in range(total_num)
            ]
        else:
            assert mock_inference_session.run_with_iobinding.call_count == total_num
            assert mock_inference_session.run_with_iobinding.mock_calls == [
                mock.call(io_bind_op) for _ in range(total_num)
            ]

    @patch("olive.systems.utils.create_new_system")
    def test_create_new_system_with_cache(self, mock_create_new_system):
        from olive.systems.utils import create_new_system_with_cache

        origin_system = PythonEnvironmentSystem(olive_managed_env=True)
        create_new_system_with_cache(origin_system, DEFAULT_CPU_ACCELERATOR)
        create_new_system_with_cache(origin_system, DEFAULT_CPU_ACCELERATOR)
        assert mock_create_new_system.call_count == 1
        create_new_system_with_cache.cache_clear()
        assert create_new_system_with_cache.cache_info().currsize == 0
