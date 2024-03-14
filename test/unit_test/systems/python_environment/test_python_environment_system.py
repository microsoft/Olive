# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import shutil
import tempfile
import venv
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
from olive.systems.python_environment.evaluation_runner import main as evaluation_runner_main
from olive.systems.python_environment.pass_runner import main as pass_runner_main
from olive.systems.system_config import PythonEnvironmentTargetUserConfig, SystemConfig
from olive.systems.utils import create_new_system, create_new_system_with_cache

# pylint: disable=no-value-for-parameter, attribute-defined-outside-init, protected-access


class TestPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        # create a virtual environment with no packages installed
        venv_path = tmp_path / "venv"
        venv.create(venv_path, with_pip=True)
        # python path
        if platform.system() == "Windows":
            self.python_environment_path = Path(venv_path) / "Scripts"
        else:
            self.python_environment_path = Path(venv_path) / "bin"
        # use the current python environment as the test environment
        self.system = PythonEnvironmentSystem(self.python_environment_path)
        yield
        shutil.rmtree(venv_path)

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

        mock_run_subprocess.return_value = (0, "test", "")

        # execute
        res = self.system._run_command(script_path, config_jsons, **extra_args)

        # assert
        assert res == dummy_output
        python_path = shutil.which("python", path=self.python_environment_path)
        expected_command = [
            python_path,
            str(script_path),
            "--dummy_config",
            str(tmp_path / "dummy_config.json"),
            "--dummy_arg",
            "dummy_arg_value",
            "--output_path",
            str(tmp_path / "output.json"),
        ]
        mock_run_subprocess.assert_called_once_with(expected_command, env=self.system.environ, check=True)
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
        dummy_pass_config = {
            "type": "DummyPass",
            "config": {
                "dummy_param_1": "dummy_param_1_value",
                "dummy_param_2": "dummy_param_2_value",
            },
        }
        full_config = {
            "dummy_param_1": "dummy_param_1_value",
            "dummy_param_2": "dummy_param_2_value2",
        }
        expected_pass_config = {"type": "DummyPass", "config": full_config}
        the_pass.to_json.return_value = dummy_pass_config
        the_pass.serialize_config.return_value = full_config

        # mock return value
        mock_return_value = {"dummy_output_model_key": "dummy_output_model_value"}
        mock__run_command.return_value = mock_return_value

        mock_output_model_config = MagicMock()
        mock_model_config_parse_obj.return_value = mock_output_model_config

        dummy_output_model_path = "dummy_output_model_path"

        # execute
        res = self.system.run_pass(the_pass, model_config, None, dummy_output_model_path)

        # assert
        assert res == mock_output_model_config
        mock_model_config_parse_obj.assert_called_once_with(mock_return_value)
        mock__run_command.assert_called_once_with(
            self.system.pass_runner_path,
            {"model_config": dummy_model_config, "pass_config": expected_pass_config},
            data_root=None,
            tempdir=tempfile.tempdir,
            output_model_path=dummy_output_model_path,
        )

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.evaluate")
    def test_evaluation_runner(self, mock_evaluate, tmp_path):
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
        config = the_pass.config_at_search_point({})
        pass_config = the_pass.to_json(check_object=True)
        pass_config["config"].update(the_pass.serialize_config(config, check_object=True))
        with (tmp_path / "pass_config.json").open("w") as f:
            json.dump(pass_config, f)

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

    @patch("olive.systems.utils.misc.create_new_system")
    def test_create_new_system_with_cache(self, mock_create_new_system):
        system_config = SystemConfig(
            type="PythonEnvironment",
            config=PythonEnvironmentTargetUserConfig(
                olive_managed_env=True,
            ),
        )
        create_new_system_with_cache(system_config, DEFAULT_CPU_ACCELERATOR)
        create_new_system_with_cache(system_config, DEFAULT_CPU_ACCELERATOR)
        assert mock_create_new_system.call_count == 1
        create_new_system_with_cache.cache_clear()
        assert create_new_system_with_cache.cache_info().currsize == 0

    def test_create_managed_env(self):
        system_config = SystemConfig(
            type="PythonEnvironment",
            config=PythonEnvironmentTargetUserConfig(
                olive_managed_env=True,
            ),
        )
        system = create_new_system(system_config, DEFAULT_CPU_ACCELERATOR)
        assert system.olive_managed_env

    def test_python_system_config(self):
        config = {
            "type": "PythonEnvironment",
            "config": {
                "python_environment_path": self.python_environment_path,
                "olive_managed_env": False,
            },
        }
        system_config = SystemConfig.parse_obj(config)
        system = system_config.create_system()
        assert system


class TestPythonEnvironmentSystemConfig:
    def test_missing_python_environment_path(self):
        invalid_config = {
            "type": "PythonEnvironment",
            "config": {
                "olive_managed_env": False,
            },
        }
        with pytest.raises(
            ValueError, match="python_environment_path is required for PythonEnvironmentSystem native mode"
        ):
            SystemConfig.parse_obj(invalid_config)

    def test_invalid_python_environment_path(self):
        invalid_config = {
            "type": "PythonEnvironment",
            "config": {
                "python_environment_path": "invalid_path",
                "olive_managed_env": False,
            },
        }
        with pytest.raises(ValueError, match="Python path invalid_path does not exist"):
            SystemConfig.parse_obj(invalid_config)

    def test_managed_system_config(self):
        config = {
            "type": "PythonEnvironment",
            "config": {
                "olive_managed_env": True,
            },
        }
        system_config = SystemConfig.parse_obj(config)
        assert system_config.config.olive_managed_env
        assert system_config.config.python_environment_path is None
