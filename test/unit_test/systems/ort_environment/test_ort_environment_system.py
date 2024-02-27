# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import sys
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_onnx_model_config
from unittest import mock
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch

from olive.constants import Framework
from olive.evaluator.metric import AccuracySubType, LatencySubType
from olive.evaluator.olive_evaluator import OliveEvaluator, OnnxEvaluator
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.systems.ort_environment import ORTEnvironmentSystem
from olive.systems.ort_environment.inference_runner import main as inference_runner_main
from olive.systems.ort_environment.ort_environment_system import ORTEnvironmentEvaluator

# pylint: disable=attribute-defined-outside-init, protected-access


def get_inference_system():
    # use the current python environment as the test environment
    executable_parent = Path(sys.executable).parent.resolve().as_posix()
    return ORTEnvironmentSystem(executable_parent)


class TestORTEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = get_inference_system()

    def test_get_supported_execution_providers(self):
        import onnxruntime as ort

        assert set(self.system.get_supported_execution_providers()) == set(ort.get_available_providers())

    def test_unsupported_pass_run(self):
        with pytest.raises(NotImplementedError):
            self.system.run_pass(None, None, None, None)

    @patch("olive.systems.ort_environment.ort_environment_system.ORTEnvironmentEvaluator.evaluate")
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

        assert "ORTEnvironmentSystem only supports evaluation for ONNXModel" in str(errinfo.value)


class TestORTEnvironmentEvaluator:
    @pytest.fixture(autouse=True)
    def setup(self):
        self.system = get_inference_system()
        self.evaluator = ORTEnvironmentEvaluator(self.system.environ)
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

    @pytest.mark.parametrize("mode", ["inference", "latency"])
    @patch("olive.systems.ort_environment.inference_runner.get_ort_inference_session")
    @patch("olive.systems.ort_environment.inference_runner.OrtInferenceSession")
    def test_inference_runner_with_run(self, mock_wrapper_class, mock_get_session, tmp_path, mode):
        # setup
        mock_wrapper = MagicMock()
        mock_wrapper_class.return_value = mock_wrapper
        if mode == "inference":
            num_batches = 3
            dummy_output = np.array([1, 2])
            mock_wrapper.run.return_value = dummy_output
        else:
            num_runs = 4
            num_warmup = 2
            sleep_time = 0
            dummy_latencies = [1, 2, 3, 4]
            mock_wrapper.time_run.return_value = dummy_latencies

        model = "model.onnx"

        # create config
        config = {
            "mode": "inference",
            "inference_settings": {
                "execution_provider": ["CPUExecutionProvider"],
            },
        }
        if mode == "inference":
            config["num_batches"] = num_batches
        else:
            config["mode"] = "latency"
            config["warmup_num"] = num_warmup
            config["repeat_test_num"] = num_runs
            config["sleep_num"] = sleep_time
        config_path = tmp_path / "config.json"
        with config_path.open("w") as f:
            json.dump(config, f)

        # create input and output directories
        input_dir = tmp_path / "input"
        input_dir.mkdir(parents=True, exist_ok=True)
        if mode == "inference":
            for i in range(num_batches):
                np.savez(input_dir / f"input_{i}.npz", input=np.array([i]))
        else:
            np.savez(input_dir / "input.npz", input=np.array([0]))
        output_dir = tmp_path / "output"
        output_dir.mkdir(parents=True, exist_ok=True)

        # command args
        args = [
            "--config_path",
            str(config_path),
            "--model_path",
            model,
            "--input_dir",
            str(input_dir),
            "--output_dir",
            str(output_dir),
        ]

        # execute
        inference_runner_main(args)

        # assert
        mock_get_session.assert_called_once_with(Path(model), config["inference_settings"], False)
        mock_wrapper_class.assert_called_once_with(
            mock_get_session.return_value,
            io_bind=False,
            device="cpu",
            shared_kv_buffer=False,
            use_fp16=False,
            input_feed={"input": np.array([0])},
        )
        if mode == "inference":
            assert mock_wrapper.run.call_count == num_batches
            assert mock_wrapper.run.mock_calls == [mock.call({"input": np.array([i])}) for i in range(num_batches)]
            for i in range(num_batches):
                batch_output_path = output_dir / f"output_{i}.npy"
                assert batch_output_path.exists()
                assert np.array_equal(np.load(batch_output_path), dummy_output)
        else:
            assert mock_wrapper.time_run.call_count == 1
            assert mock_wrapper.time_run.mock_calls == [
                mock.call({"input": np.array([0])}, num_runs=num_runs, num_warmup=num_warmup, sleep_time=sleep_time)
            ]
            assert (output_dir / "output.npy").exists()
            assert np.array_equal(np.load(output_dir / "output.npy"), np.array(dummy_latencies))
