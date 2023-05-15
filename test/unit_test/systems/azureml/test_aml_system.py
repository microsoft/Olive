# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import tempfile
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_latency_metric, get_pytorch_model
from unittest.mock import MagicMock, Mock, patch

import pytest
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType
from olive.model import ModelStorageKind, ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.systems.azureml.aml_system import AzureMLSystem
from olive.systems.common import AzureMLDockerConfig


class TestAzureMLSystem:
    @pytest.fixture(autouse=True)
    @patch("olive.systems.azureml.aml_system.Environment")
    def setup(self, mock_env):
        docker_config = AzureMLDockerConfig(
            base_image="base_image",
            conda_file_path="conda_file_path",
        )
        mock_azureml_client_config = Mock(spec=AzureMLClientConfig)
        self.system = AzureMLSystem(mock_azureml_client_config, "dummy", docker_config)

    METRIC_TEST_CASE = [
        (get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)),
        (get_accuracy_metric(AccuracySubType.F1_SCORE)),
        (get_accuracy_metric(AccuracySubType.PRECISION)),
        (get_accuracy_metric(AccuracySubType.RECALL)),
        (get_accuracy_metric(AccuracySubType.AUC)),
        (get_latency_metric(LatencySubType.AVG)),
        (get_latency_metric(LatencySubType.MAX)),
        (get_latency_metric(LatencySubType.MIN)),
        (get_latency_metric(LatencySubType.P50)),
        (get_latency_metric(LatencySubType.P75)),
        (get_latency_metric(LatencySubType.P90)),
        (get_latency_metric(LatencySubType.P95)),
        (get_latency_metric(LatencySubType.P99)),
        (get_latency_metric(LatencySubType.P999)),
    ]

    @pytest.mark.parametrize(
        "metric",
        METRIC_TEST_CASE,
    )
    @patch("olive.systems.azureml.aml_system.retry_func")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_evaluation")
    @patch("olive.systems.azureml.aml_system.tempfile.TemporaryDirectory")
    def test_evaluate_model(self, mock_tempdir, mock_create_pipeline, mock_retry_func, metric):
        # setup
        olive_model = get_pytorch_model()
        output_folder = Path(__file__).absolute().parent / "output_metrics"
        mock_tempdir.return_value.__enter__.return_value = output_folder

        # execute
        res = self.system.evaluate_model(olive_model, [metric])[metric.name]

        # assert
        mock_create_pipeline.assert_called_once_with(output_folder, olive_model, [metric])
        self.system.ml_client.jobs.stream.assert_called_once()
        assert mock_retry_func.call_count == 2
        if metric.name == "accuracy":
            assert res == 0.99618
        if metric.name == "latency":
            assert res == 0.031415

    @patch("olive.systems.azureml.aml_system.shutil.copy")
    @patch("olive.systems.azureml.aml_system.retry_func")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_pass")
    @patch("olive.systems.azureml.aml_system.tempfile.TemporaryDirectory")
    def test_run_pass(self, mock_tempdir, mock_create_pipeline, mock_retry_func, mock_copy):
        # setup
        onnx_conversion_config = {}
        p = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
        olive_model = get_pytorch_model()
        output_model_path = "output_model_path"
        output_folder = Path(__file__).absolute().parent / "output_pass"
        mock_tempdir.return_value.__enter__.return_value = output_folder
        expected_model = ONNXModel(model_path=ONNXModel.resolve_path(output_model_path), name="test_model")

        # execute
        actual_res = self.system.run_pass(p, olive_model, output_model_path)

        # assert
        mock_create_pipeline.assert_called_once_with(output_folder, olive_model, p.to_json(), p.path_params)
        assert mock_retry_func.call_count == 2
        self.system.ml_client.jobs.stream.assert_called_once()
        assert expected_model.to_json() == actual_res.to_json()

    @pytest.mark.parametrize(
        "model_storage_kind",
        [ModelStorageKind.AzureMLModel, ModelStorageKind.LocalFile],
    )
    def test__create_model_args(self, model_storage_kind):
        # setup
        temp_model = tempfile.NamedTemporaryFile(dir=".", suffix=".onnx", prefix="model_0")
        model_json = {
            "type": "onnxmodel",
            "config": {
                "model_script": "model_script",
                "script_dir": "script_dir",
                "model_path": temp_model.name,
                "model_storage_kind": model_storage_kind,
            },
        }
        tem_dir = Path(".")
        model_config_path = tem_dir / "model_config.json"
        model_path = tem_dir / "model.onnx"
        if model_storage_kind == ModelStorageKind.AzureMLModel:
            expected_model_path = Input(type=AssetTypes.CUSTOM_MODEL, path=temp_model.name)
        else:
            expected_model_path = Input(type=AssetTypes.URI_FILE, path=model_path)
        expected_model_script = Input(type=AssetTypes.URI_FILE, path="model_script")
        expected_model_script_dir = Input(type=AssetTypes.URI_FOLDER, path="script_dir")
        expected_model_config = Input(type=AssetTypes.URI_FILE, path=model_config_path)
        expected_res = {
            "model_config": expected_model_config,
            "model_path": expected_model_path,
            "model_script": expected_model_script,
            "model_script_dir": expected_model_script_dir,
        }

        # execute
        actual_res = self.system._create_model_args(model_json, tem_dir)

        # assert
        assert actual_res == expected_res
        assert model_json["config"]["model_script"] is None
        assert model_json["config"]["script_dir"] is None
        assert model_json["config"]["model_storage_kind"] != ModelStorageKind.AzureMLModel
        assert model_json["config"]["model_path"] is None

        # cleanup
        if os.path.exists(model_path):
            os.remove(model_path)
        if os.path.exists(model_config_path):
            os.remove(model_config_path)

    def test__create_metric_args(self):
        # setup
        tem_dir = Path(__file__).absolute().parent
        metric_config = {
            "user_config": {
                "user_script": "user_script",
                "script_dir": "script_dir",
                "data_dir": "data_dir",
            }
        }
        metric_config_path = tem_dir / "metric_config.json"

        expected_metric_config = Input(type=AssetTypes.URI_FILE, path=metric_config_path)
        expected_metric_user_script = Input(type=AssetTypes.URI_FILE, path="user_script")
        expected_metric_script_dir = Input(type=AssetTypes.URI_FOLDER, path="script_dir")
        expected_metric_data_dir = Input(type=AssetTypes.URI_FOLDER, path="data_dir")
        expected_res = {
            "metric_config": expected_metric_config,
            "metric_user_script": expected_metric_user_script,
            "metric_script_dir": expected_metric_script_dir,
            "metric_data_dir": expected_metric_data_dir,
        }

        # execute
        actual_res = self.system._create_metric_args(metric_config, tem_dir)

        # assert
        assert actual_res == expected_res
        assert metric_config["user_config"]["user_script"] is None
        assert metric_config["user_config"]["script_dir"] is None
        assert metric_config["user_config"]["data_dir"] is None

        # cleanup
        if os.path.exists(metric_config_path):
            os.remove(metric_config_path)

    @pytest.mark.parametrize(
        "model_storage_kind",
        [ModelStorageKind.AzureMLModel, ModelStorageKind.LocalFile],
    )
    @patch("olive.systems.azureml.aml_system.shutil.copy")
    @patch("olive.systems.azureml.aml_system.command")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_metric_args")
    def test__create_metric_component(self, mock_create_metric_args, mock_command, mock_copy, model_storage_kind):
        # setup
        tem_dir = Path(".")
        code_path = tem_dir / "code"
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        metric.user_config = {}
        model_args = {"input": Input(type=AssetTypes.URI_FILE, path="path")}
        metric_type = f"{metric.type}-{metric.sub_type}"
        model_inputs = {
            "model_config": Input(type=AssetTypes.URI_FILE),
            "model_path": Input(type=AssetTypes.CUSTOM_MODEL)
            if model_storage_kind == ModelStorageKind.AzureMLModel
            else Input(type=AssetTypes.URI_FOLDER, optional=True),
            "model_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "model_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }
        metric_inputs = {
            "metric_config": Input(type=AssetTypes.URI_FILE),
            "metric_user_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "metric_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
            "metric_data_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }
        inputs = {**model_inputs, **metric_inputs}
        expected_res = MagicMock()
        mock_command.return_value.return_value = expected_res

        # execute
        actual_res = self.system._create_metric_component(tem_dir, metric, model_args, model_storage_kind)

        # assert
        assert actual_res == expected_res
        mock_command.assert_called_once_with(
            name=metric_type,
            display_name=metric_type,
            description=f"Run olive {metric_type} evaluation",
            command=self.create_command(inputs),
            environment=self.system.environment,
            code=code_path,
            inputs=inputs,
            outputs=dict(pipeline_output=Output(type=AssetTypes.URI_FOLDER)),
            instance_count=1,
            compute=self.system.compute,
        )

        # cleanup
        if os.path.exists(code_path):
            os.rmdir(code_path)

    def create_command(self, inputs):
        script_name = "aml_evaluation_runner.py"
        parameters = []
        for param, input in inputs.items():
            if input.optional:
                parameters.append(f"$[[--{param} ${{{{inputs.{param}}}}}]]")
            else:
                parameters.append(f"--{param} ${{{{inputs.{param}}}}}")
        parameters.append("--pipeline_output ${{outputs.pipeline_output}}")

        return f"python {script_name} {' '.join(parameters)}"
