# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import pickle
import sys
import tempfile
from pathlib import Path
from test.unit_test.utils import (
    get_glue_accuracy_metric,
    get_glue_latency_metric,
    get_hf_model_with_past,
    get_onnx_model,
    get_onnxconversion_pass,
)
from unittest.mock import MagicMock, patch

import pytest

from olive.evaluator.metric import MetricResult, joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ModelConfig
from olive.systems.python_environment import PythonEnvironmentSystem
from olive.systems.python_environment.available_eps import main as available_eps_main
from olive.systems.python_environment.evaluation_runner import main as evaluation_runner_main
from olive.systems.python_environment.pass_runner import main as pass_runner_main

# pylint: disable=no-value-for-parameter, attribute-defined-outside-init


class TestPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the current python environment as the test environment
        executable_parent = Path(sys.executable).parent.resolve().as_posix()
        self.system = PythonEnvironmentSystem(executable_parent)

    def test_get_supported_execution_providers(self):
        import onnxruntime as ort

        assert set(self.system.get_supported_execution_providers()) == set(ort.get_available_providers())

    @patch("olive.systems.python_environment.python_environment_system.run_subprocess")
    @patch("olive.systems.python_environment.python_environment_system.tempfile.TemporaryDirectory")
    def test__run_command(self, mock_temp_dir, mock_run_subprocess, tmp_path):
        # setup
        mock_temp_dir.return_value = tmp_path.resolve()
        # input
        script_path = "dummy_script.py"
        config_jsons = {"dummy_config": {"dummy_key": "dummy_value"}}
        extra_args = {"dummy_arg": "dummy_arg_value"}
        # output
        dummy_output_path = tmp_path / "output.json"
        dummy_output = {"dummy_out_key": "dummy_out_value"}
        with dummy_output_path.open("w") as f:
            json.dump(dummy_output, f)

        mock_run_subprocess.return_value = None

        # execute
        res = self.system._run_command(script_path, config_jsons, **extra_args)

        # assert
        assert res == dummy_output
        mock_run_subprocess.assert_called_once_with(
            f"python {script_path} --dummy_config {tmp_path / 'dummy_config.json'} --dummy_arg dummy_arg_value "
            f"--output_path {tmp_path / 'output.json'}",
            env=self.system.environ,
            check=True,
        )
        with (tmp_path / "dummy_config.json").open("r") as f:
            assert json.load(f) == config_jsons["dummy_config"]

    @patch("olive.systems.python_environment.python_environment_system.PythonEnvironmentSystem._run_command")
    def test_evaluate_model(self, mock__run_command):
        # setup
        model_config = MagicMock()
        dummy_model_config = {"dummy_key": "dummy_value"}
        model_config.to_json.return_value = dummy_model_config
        metrics = [get_glue_accuracy_metric(), get_glue_latency_metric()]

        # mock return value
        metrics_key = [
            joint_metric_key(metric.name, sub_metric.name) for metric in metrics for sub_metric in metric.sub_types
        ]
        metrics_values = [
            {"value": 0.9, "priority": 1, "higher_is_better": True},
            {"value": 10, "priority": 2, "higher_is_better": False},
        ]
        mock_return_value = dict(key_value for key_value in zip(metrics_key, metrics_values))
        mock__run_command.return_value = mock_return_value

        # execute
        res = self.system.evaluate_model(model_config, None, metrics, DEFAULT_CPU_ACCELERATOR)

        # assert
        assert res[metrics_key[0]].value == 0.9
        assert res[metrics_key[1]].value == 10
        mock__run_command.assert_called_once_with(
            self.system.evaluation_runner_path,
            {
                "model_config": dummy_model_config,
                "metrics_config": [metric.to_json() for metric in metrics],
                "accelerator_config": DEFAULT_CPU_ACCELERATOR.to_json(),
            },
            data_root=None,
            tempdir=tempfile.tempdir,
        )

    @patch("olive.systems.python_environment.python_environment_system.PythonEnvironmentSystem._run_command")
    @patch("olive.systems.python_environment.python_environment_system.ModelConfig.parse_obj")
    def test_run_pass(self, mock_model_config_parse_obj, mock__run_command):
        # setup
        model_config = MagicMock()
        dummy_model_config = {"dummy_model_key": "dummy_model_value"}
        model_config.to_json.return_value = dummy_model_config
        the_pass = MagicMock()
        dummy_the_pass_config = {"dummy_pass_key": "dummy_pass_value"}
        the_pass.to_json.return_value = dummy_the_pass_config
        point = {"dummy_point_key": "dummy_point_value"}

        # mock return value
        mock_return_value = {"dummy_output_model_key": "dummy_output_model_value"}
        mock__run_command.return_value = mock_return_value

        mock_output_model_config = MagicMock()
        mock_model_config_parse_obj.return_value = mock_output_model_config

        dummy_output_model_path = "dummy_output_model_path"

        # execute
        res = self.system.run_pass(the_pass, model_config, None, dummy_output_model_path, point)

        # assert
        assert res == mock_output_model_config
        mock_model_config_parse_obj.assert_called_once_with(mock_return_value)
        mock__run_command.assert_called_once_with(
            self.system.pass_runner_path,
            {
                "model_config": dummy_model_config,
                "pass_config": dummy_the_pass_config,
                "point_config": point,
            },
            data_root=None,
            tempdir=tempfile.tempdir,
            output_model_path=dummy_output_model_path,
        )

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.evaluate")
    def test_aml_evaluation_runner(self, mock_evaluate, tmp_path):
        # create model_config.json
        model_config = ModelConfig.parse_obj(get_onnx_model().to_json()).to_json()
        with (tmp_path / "model_config.json").open("w") as f:
            json.dump(model_config, f)

        # create metrics_config.json
        metrics_config = [get_glue_accuracy_metric().to_json()]
        with (tmp_path / "metrics_config.json").open("w") as f:
            json.dump(metrics_config, f)

        # create accelerator_config.json
        accelerator_config = DEFAULT_CPU_ACCELERATOR.to_json()
        with (tmp_path / "accelerator_config.json").open("w") as f:
            json.dump(accelerator_config, f)

        ouptut_path = tmp_path / "output.json"

        # mock output
        mock_evaluation_result = MetricResult.parse_obj(
            {"accuracy-accuracy_score": {"value": 0.5, "priority": 1, "higher_is_better": True}}
        )
        mock_evaluate.return_value = mock_evaluation_result

        # execute
        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--metrics_config",
            str(tmp_path / "metrics_config.json"),
            "--accelerator_config",
            str(tmp_path / "accelerator_config.json"),
            "--output_path",
            str(ouptut_path),
        ]
        evaluation_runner_main(args)

        # assert
        mock_evaluate.assert_called_once()
        with ouptut_path.open("r") as f:
            assert json.load(f) == mock_evaluation_result.to_json()

    @patch("olive.passes.onnx.conversion.OnnxConversion.run")
    def test_pass_runner(self, mock_conversion_run, tmp_path):
        # create model_config.json
        model_config = ModelConfig.parse_obj(get_hf_model_with_past().to_json()).to_json()
        with (tmp_path / "model_config.json").open("w") as f:
            json.dump(model_config, f)

        # create pass_config.json
        the_pass = get_onnxconversion_pass()
        pass_config = the_pass.to_json()
        with (tmp_path / "pass_config.json").open("w") as f:
            json.dump(pass_config, f)

        # create point_config.json
        with (tmp_path / "point_config.json").open("w") as f:
            json.dump({}, f)

        output_path = tmp_path / "output.json"

        # mock output
        dummy_output_model_path = "dummy_output_model_path"
        mock_conversion_run.return_value = get_onnx_model()

        # execute
        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--pass_config",
            str(tmp_path / "pass_config.json"),
            "--point_config",
            str(tmp_path / "point_config.json"),
            "--output_model_path",
            dummy_output_model_path,
            "--output_path",
            str(output_path),
        ]
        pass_runner_main(args)

        # assert
        mock_conversion_run.assert_called_once()
        with output_path.open("r") as f:
            assert json.load(f) == get_onnx_model().to_json()

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

    @patch("olive.systems.utils.create_new_system")
    def test_create_new_system_with_cache(self, mock_create_new_system):
        from olive.systems.utils import create_new_system_with_cache

        origin_system = PythonEnvironmentSystem(olive_managed_env=True)
        create_new_system_with_cache(origin_system, DEFAULT_CPU_ACCELERATOR)
        create_new_system_with_cache(origin_system, DEFAULT_CPU_ACCELERATOR)
        assert mock_create_new_system.call_count == 1
        create_new_system_with_cache.cache_clear()
        assert create_new_system_with_cache.cache_info().currsize == 0
