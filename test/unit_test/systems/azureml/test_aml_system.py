# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import tempfile
from functools import partial
from pathlib import Path
from test.unit_test.utils import (
    ONNX_MODEL_PATH,
    get_accuracy_metric,
    get_custom_metric,
    get_glue_latency_metric,
    get_hf_model_config,
    get_latency_metric,
    get_onnx_model_config,
    get_onnxconversion_pass,
)
from typing import ClassVar, List
from unittest.mock import ANY, MagicMock, Mock, patch

import pytest
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.entities import UserIdentityConfiguration
from azure.core.exceptions import ResourceNotFoundError

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.constants import HF_LOGIN, KEYVAULT_NAME
from olive.data.config import DataConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, Metric
from olive.evaluator.metric_result import MetricResult
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.resource_path import ResourcePath, ResourceType
from olive.systems.azureml.aml_evaluation_runner import main as aml_evaluation_runner_main
from olive.systems.azureml.aml_pass_runner import main as aml_pass_runner_main
from olive.systems.azureml.aml_system import AzureMLSystem
from olive.systems.common import AzureMLDockerConfig, AzureMLEnvironmentConfig

# pylint: disable=attribute-defined-outside-init, protected-access


class TestAzureMLSystem:
    @pytest.fixture(autouse=True)
    @patch("olive.systems.azureml.aml_system.Environment")
    def setup(self, mock_env):
        docker_config = AzureMLDockerConfig(
            base_image="base_image",
            conda_file_path="conda_file_path",
        )
        mock_azureml_client_config = Mock(spec=AzureMLClientConfig)
        self.system = AzureMLSystem(mock_azureml_client_config, "dummy", docker_config, is_dev=True)

        self.input_model_config = {
            "config": {
                "model_path": "hf-internal-testing/tiny-random-BertForSequenceClassification",
                "task": "text-classification",
                "io_config": {
                    "dynamic_axes": {
                        "attention_mask": {"0": "batch_size", "1": "seq_length"},
                        "input_ids": {"0": "batch_size", "1": "seq_length"},
                        "token_type_ids": {"0": "batch_size", "1": "seq_length"},
                    },
                    "input_names": ["input_ids", "attention_mask", "token_type_ids"],
                    "input_shapes": [[1, 128], [1, 128], [1, 128]],
                    "input_types": ["int64", "int64", "int64"],
                    "output_names": ["output"],
                },
            },
            "type": "HfModel",
        }

    METRIC_TEST_CASE: ClassVar[List[Metric]] = [
        (partial(get_accuracy_metric, AccuracySubType.ACCURACY_SCORE)),
        (partial(get_accuracy_metric, AccuracySubType.F1_SCORE)),
        (partial(get_accuracy_metric, AccuracySubType.PRECISION)),
        (partial(get_accuracy_metric, AccuracySubType.RECALL)),
        (partial(get_accuracy_metric, AccuracySubType.AUROC)),
        (partial(get_latency_metric, LatencySubType.AVG)),
        (partial(get_latency_metric, LatencySubType.MAX)),
        (partial(get_latency_metric, LatencySubType.MIN)),
        (partial(get_latency_metric, LatencySubType.P50)),
        (partial(get_latency_metric, LatencySubType.P75)),
        (partial(get_latency_metric, LatencySubType.P90)),
        (partial(get_latency_metric, LatencySubType.P95)),
        (partial(get_latency_metric, LatencySubType.P99)),
        (partial(get_latency_metric, LatencySubType.P999)),
    ]

    @pytest.mark.parametrize(
        "metric_func",
        METRIC_TEST_CASE,
    )
    @patch("olive.systems.azureml.aml_system.retry_func")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_evaluation")
    @patch("olive.systems.azureml.aml_system.tempfile.TemporaryDirectory")
    def test_evaluate_model(self, mock_tempdir, mock_create_pipeline, mock_retry_func, metric_func):
        # setup
        model_config = get_onnx_model_config()
        output_folder = Path(__file__).absolute().parent / "output_metrics"
        mock_tempdir.return_value.__enter__.return_value = output_folder
        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client
        self.system.azureml_client_config.max_operation_retries = 3
        self.system.azureml_client_config.operation_retry_interval = 5

        # execute
        metric = metric_func()
        res = self.system.evaluate_model(model_config, [metric], DEFAULT_CPU_ACCELERATOR)

        # assert
        mock_create_pipeline.assert_called_once_with(
            output_folder, model_config.to_json(check_object=True), [metric], DEFAULT_CPU_ACCELERATOR
        )
        ml_client.jobs.stream.assert_called_once()
        assert mock_retry_func.call_count == 2
        if metric.name == "accuracy":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.99618
        if metric.name == "latency":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.031415

    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_workflow")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._run_job")
    def test_submit_workflow(self, mock_run_job, mock_create_pipeline_for_workflow):
        # setup
        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client
        run_config = MagicMock(workflow_id="workflow_id")

        # execute
        self.system.submit_workflow(run_config)

        # assert
        mock_create_pipeline_for_workflow.assert_called_once()
        mock_run_job.assert_called_once()

    @patch("olive.systems.azureml.aml_system.retry_func")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_pass")
    def test_run_pass(self, mock_create_pipeline, mock_retry_func, tmp_path):
        # setup
        # dummy pipeline output download path
        pipeline_output_path = tmp_path / "pipeline_output" / "named-outputs" / "pipeline_output"
        pipeline_output_path.mkdir(parents=True, exist_ok=True)
        # create dummy output model
        downloaded_output_model_path = pipeline_output_path / "output_model.onnx"
        with downloaded_output_model_path.open("w") as f:
            f.write("dummy")
        # create dummy output config
        dummy_config = {
            "type": "ONNXModel",
            "config": {
                "model_path": {"type": "file", "config": {"path": "output_model.onnx"}},
                "inference_settings": None,
            },
            "same_resources_as_input": [],
            "resources": [["config", "model_path"]],
        }
        dummy_config_path = pipeline_output_path / "output_model_config.json"
        with dummy_config_path.open("w") as f:
            json.dump(dummy_config, f, indent=4)

        onnx_conversion_config = {}
        p = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
        model_config = get_hf_model_config()
        output_model_path = tmp_path / "output_folder" / "output_model_path"
        output_model_path.mkdir(parents=True, exist_ok=True)
        # create dummy output model so that ONNXModel can be created with the same path
        expected_model_path = output_model_path / "model.onnx"
        with expected_model_path.open("w") as f:
            f.write("dummy")
        output_folder = tmp_path

        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client
        self.system.azureml_client_config.max_operation_retries = 3
        self.system.azureml_client_config.operation_retry_interval = 5

        with patch("olive.systems.azureml.aml_system.tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir.return_value.__enter__.return_value = output_folder
            # execute
            actual_res = self.system.run_pass(p, model_config, output_model_path)

        # assert
        mock_create_pipeline.assert_called_once_with(
            output_folder, model_config.to_json(check_object=True), p.to_json()
        )
        assert mock_retry_func.call_count == 2
        ml_client.jobs.stream.assert_called_once()
        output_model_file = actual_res.config["model_path"]
        if isinstance(output_model_file, ResourcePath):
            output_model_file = output_model_file.get_path()
        assert Path(output_model_file).samefile(expected_model_path)

    @pytest.mark.parametrize(
        "model_resource_type",
        [ResourceType.AzureMLModel, ResourceType.LocalFile, ResourceType.StringName],
    )
    def test_model_inputs_and_args(self, model_resource_type, tmp_path):
        # setup
        ws_config = {
            "workspace_name": "workspace_name",
            "subscription_id": "subscription_id",
            "resource_group": "resource_group",
        }
        self.system.azureml_client_config.get_workspace_config.return_value = ws_config
        with tempfile.NamedTemporaryFile(dir=".", suffix=".onnx", prefix="model_0") as temp_model:
            resource_paths = {
                ResourceType.AzureMLModel: {
                    "type": ResourceType.AzureMLModel,
                    "config": {
                        "azureml_client": ws_config,
                        "name": "model_name",
                        "version": "version",
                    },
                },
                ResourceType.LocalFile: temp_model.name,
                ResourceType.StringName: "model_name",
            }
            model_json = {
                "type": "pytorchmodel",
                "config": {
                    "model_path": resource_paths[model_resource_type],
                    "model_attributes": {
                        "_model_name_or_path": resource_paths[model_resource_type],
                    },
                    # this won't happen in real life, but we want to test the case
                    # where the same resource is used in multiple places
                    "model_path2": resource_paths[model_resource_type],
                },
            }

            expected_inputs = {
                "model_config": Input(type=AssetTypes.URI_FILE),
                "model_resource_map": Input(type=AssetTypes.URI_FILE),
                "num_resources": Input(type="integer"),
            }
            expected_args = {
                "model_config": Input(type=AssetTypes.URI_FILE, path=tmp_path / "model_config.json"),
                "model_resource_map": Input(type=AssetTypes.URI_FILE, path=tmp_path / "model_resource_map.json"),
            }

            if model_resource_type == ResourceType.AzureMLModel:
                expected_args["num_resources"] = 1
                expected_inputs["resource__0"] = Input(type=AssetTypes.CUSTOM_MODEL)
                expected_args["resource__0"] = Input(type=AssetTypes.CUSTOM_MODEL, path="azureml:model_name:version")
            elif model_resource_type == ResourceType.LocalFile:
                expected_args["num_resources"] = 1
                expected_inputs["resource__0"] = Input(type=AssetTypes.URI_FILE)
                expected_args["resource__0"] = Input(type=AssetTypes.URI_FILE, path=Path(temp_model.name).resolve())
            else:
                del expected_inputs["model_resource_map"], expected_args["model_resource_map"]
                expected_args["num_resources"] = 0

            # execute
            actual_inputs, actual_args = self.system.create_inputs_and_args(
                {"model": model_json}, tmp_path, ignore_keys=["model_attributes"]
            )

        # assert
        assert actual_inputs == expected_inputs
        assert actual_args == expected_args

        with open(tmp_path / "model_config.json") as f:
            model_json = json.load(f)

            if model_resource_type != ResourceType.StringName:
                assert model_json["config"]["model_path"] is None
                assert model_json["config"]["model_path2"] is None
            else:
                assert model_json["config"]["model_path"] == resource_paths[model_resource_type]
                assert model_json["config"]["model_path2"] == resource_paths[model_resource_type]

        if model_resource_type != ResourceType.StringName:
            with open(tmp_path / "model_resource_map.json") as f:
                resource_map = json.load(f)
                assert resource_map == [
                    [["config", "model_path"], "resource__0"],
                    [["config", "model_path2"], "resource__0"],
                ]

    @patch("olive.systems.azureml.aml_system.command")
    def test__create_step(self, mock_command):
        # setup
        name = "name"
        display_name = "display_name"
        description = "description"
        aml_environment = MagicMock()
        code = "code"
        compute = "compute"
        instance_count = 1
        inputs = {"dummy_input": Input(type=AssetTypes.URI_FILE)}
        outputs = {"dummy_output": Output(type=AssetTypes.URI_FILE)}
        script_name = "aml_evaluation_runner.py"
        resources = {
            "instance_type": "instance_type",
            "properties": {
                "AISuperComputer": {
                    "interactive": True,
                    "imageVersion": "imageVersion",
                    "slaTier": "slaTier",
                    "priority": "priority",
                    "tensorboardLogDirectory": "tensorboardLogDirectory",
                    "enableAzmlInt": True,
                }
            },
        }

        expected_res = MagicMock()
        mock_command.return_value = expected_res

        # execute
        actual_res = self.system._create_step(
            name,
            display_name,
            description,
            aml_environment,
            code,
            compute,
            resources,
            instance_count,
            inputs,
            outputs,
            script_name,
        )

        # assert
        assert actual_res == expected_res
        mock_command.assert_called_once_with(
            name=name,
            display_name=display_name,
            description=description,
            command=self.create_command(script_name, inputs, outputs),
            resources=resources,
            environment=aml_environment,
            environment_variables=ANY,
            code=code,
            inputs=inputs,
            outputs=outputs,
            instance_count=1,
            compute=compute,
            identity=UserIdentityConfiguration(),
        )

    def test_pass_inputs_and_args(self, tmp_path):
        # setup
        script_dir_path = Path(__file__).absolute().parent / "script_dir"
        user_script_path = script_dir_path / "user_script.py"
        data_dir_path = Path(__file__).absolute().parent / "data_dir"
        data_files_path = data_dir_path / "datafile.json"
        data_config = DataConfig(
            type="HuggingfaceContainer",
            name="data_name",
            user_script=str(user_script_path),
            script_dir=str(script_dir_path),
            load_dataset_config={"params": {"data_dir": data_dir_path, "data_files": data_files_path}},
        )
        pass_config = {"data_config": data_config.to_json()}

        expected_inputs = {
            "pass_config": Input(type=AssetTypes.URI_FILE),
            "pass_resource_map": Input(type=AssetTypes.URI_FILE),
            "num_resources": Input(type="integer"),
            "resource__0": Input(type=AssetTypes.URI_FILE),
            "resource__1": Input(type=AssetTypes.URI_FOLDER),
            "resource__2": Input(type=AssetTypes.URI_FOLDER),
            "resource__3": Input(type=AssetTypes.URI_FILE),
        }
        expected_args = {
            "pass_config": Input(type=AssetTypes.URI_FILE, path=(tmp_path / "pass_config.json").resolve()),
            "pass_resource_map": Input(type=AssetTypes.URI_FILE, path=(tmp_path / "pass_resource_map.json").resolve()),
            "num_resources": 4,
            "resource__0": Input(type=AssetTypes.URI_FILE, path=str(user_script_path)),
            "resource__1": Input(type=AssetTypes.URI_FOLDER, path=str(script_dir_path)),
            "resource__2": Input(type=AssetTypes.URI_FOLDER, path=str(data_dir_path)),
            "resource__3": Input(type=AssetTypes.URI_FILE, path=str(data_files_path)),
        }

        # execute
        actual_inputs, actual_args = self.system.create_inputs_and_args({"pass": pass_config}, tmp_path)

        # assert
        assert actual_inputs == expected_inputs
        # use the Path.__eq__ to compare path since sometimes in Windows, the path root
        # will be changed to uppercase, which will cause the test to fail.
        assert actual_args.keys() == expected_args.keys()
        for k, v in actual_args.items():
            if k == "num_resources":
                assert v == expected_args[k]
                continue
            assert v.type == expected_args[k].type
            assert Path(v.path) == Path(expected_args[k].path)

        with open(tmp_path / "pass_resource_map.json") as f:
            resource_map = json.load(f)
            assert resource_map == [
                [["data_config", "user_script"], "resource__0"],
                [["data_config", "script_dir"], "resource__1"],
                [["data_config", "load_dataset_config", "params", "data_dir"], "resource__2"],
                [["data_config", "load_dataset_config", "params", "data_files"], "resource__3"],
            ]

    @patch("olive.systems.azureml.aml_system.command")
    def test__create_metric_component(self, mock_command, tmp_path):
        # setup
        metric = get_custom_metric()
        sub_type_name = ",".join([st.name for st in metric.sub_types])
        metric_type = f"{metric.type}-{sub_type_name}"

        inputs = {
            "model_config": Input(type=AssetTypes.URI_FILE),
            "metric_config": Input(type=AssetTypes.URI_FILE),
            "metric_resource_map": Input(type=AssetTypes.URI_FILE),
            "accelerator_config": Input(type=AssetTypes.URI_FILE),
            "num_resources": Input(type="integer"),
            "resource__0": Input(type=AssetTypes.URI_FILE),
        }
        outputs = {"pipeline_output": Output(type=AssetTypes.URI_FOLDER)}
        expected_res = MagicMock()
        mock_command.return_value.return_value = expected_res

        # execute
        actual_res = self.system._create_metric_component(
            tmp_dir=tmp_path, model_config={}, metric=metric, accelerator_config={}
        )

        # assert
        assert actual_res == expected_res
        mock_command.assert_called_once_with(
            name=metric_type,
            display_name=metric_type,
            description=f"Run olive {metric_type} evaluation",
            command=self.create_command("aml_evaluation_runner.py", inputs, outputs),
            resources=None,
            environment=self.system.environment,
            environment_variables=ANY,
            code=str(tmp_path / "code"),
            inputs=inputs,
            outputs={"pipeline_output": Output(type=AssetTypes.URI_FOLDER)},
            instance_count=1,
            compute=self.system.compute,
            identity=UserIdentityConfiguration(),
        )

    def create_command(self, script_name, inputs, outputs):
        parameters = []
        inputs = inputs or {}
        for param, input_param in inputs.items():
            if input_param.optional:
                parameters.append(f"$[[--{param} ${{{{inputs.{param}}}}}]]")
            else:
                parameters.append(f"--{param} ${{{{inputs.{param}}}}}")
        outputs = outputs or {}
        parameters.extend([f"--{param} ${{{{outputs.{param}}}}}" for param in outputs])

        return f"python {script_name} {' '.join(parameters)}"

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.evaluate")
    def test_aml_evaluation_runner(self, mock_evaluate, tmp_path):
        # create model_config.json
        with (tmp_path / "model_config.json").open("w") as f:
            json.dump(self.input_model_config, f)

        # create model.pt
        # create metrics_config.json
        metrics_config = get_glue_latency_metric().to_json()
        with (tmp_path / "metrics_config.json").open("w") as f:
            json.dump(metrics_config, f)

        # create accelerator_config.json
        accelerator_config = DEFAULT_CPU_ACCELERATOR.to_json()
        with (tmp_path / "accelerator_config.json").open("w") as f:
            json.dump(accelerator_config, f)

        output_dir = tmp_path / "pipeline_output"
        output_dir.mkdir()

        # mock output
        mock_evaluation_result = MetricResult.parse_obj(
            {"accuracy-accuracy_score": {"value": 0.5, "priority": 1, "higher_is_better": True}}
        )
        mock_evaluate.return_value = mock_evaluation_result

        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--metric_config",
            str(tmp_path / "metrics_config.json"),
            "--accelerator_config",
            str(tmp_path / "accelerator_config.json"),
            "--num_resources",
            0,
            "--pipeline_output",
            str(output_dir),
        ]
        aml_evaluation_runner_main(args)

        # assert
        mock_evaluate.assert_called_once()
        with (output_dir / "metric_result.json").open() as f:
            assert json.load(f) == mock_evaluation_result.to_json()

    @patch("olive.passes.onnx.conversion.OnnxConversion.run")
    def test_aml_pass_runner(self, mock_conversion_run, tmp_path):
        # create model_config.json
        with (tmp_path / "model_config.json").open("w") as f:
            json.dump(self.input_model_config, f)

        # create pass_config.json
        the_pass = get_onnxconversion_pass()
        pass_config = the_pass.to_json()
        with (tmp_path / "pass_config.json").open("w") as f:
            json.dump(pass_config, f)

        output_dir = tmp_path / "pipeline_output"
        output_dir.mkdir()

        # mock output
        shutil.copy(ONNX_MODEL_PATH, output_dir)
        mock_conversion_run.return_value = ONNXModelHandler(output_dir / ONNX_MODEL_PATH.name)

        # execute
        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--pass_config",
            str(tmp_path / "pass_config.json"),
            "--num_resources",
            0,
            "--pipeline_output",
            str(output_dir),
        ]
        aml_pass_runner_main(args)

        # assert
        mock_conversion_run.assert_called_once()
        with (output_dir / "output_model_config.json").open() as f:
            output_model_json = json.load(f)
            assert output_model_json["type"] == "ONNXModel"
            assert output_model_json["config"]["model_path"]["type"] == "file"
            assert output_model_json["config"]["model_path"]["config"]["path"] == ONNX_MODEL_PATH.name


@patch("olive.systems.azureml.aml_system.Environment")
def test_aml_system_with_hf_token(mock_env):
    # setup
    mock_azureml_client_config = Mock(spec=AzureMLClientConfig)
    mock_azureml_client_config.keyvault_name = "keyvault_name"
    docker_config = AzureMLDockerConfig(
        base_image="base_image",
        conda_file_path="conda_file_path",
    )
    expected_env_vars = {HF_LOGIN: True, KEYVAULT_NAME: "keyvault_name"}

    # execute
    system = AzureMLSystem(mock_azureml_client_config, "dummy", docker_config, hf_token=True)

    # assert
    assert system.env_vars == expected_env_vars


@patch("olive.systems.azureml.aml_system.Environment")
def test_aml_system_no_keyvault_name_raise_valueerror(mock_env):
    # setup
    mock_azureml_client_config = Mock(spec=AzureMLClientConfig)
    mock_azureml_client_config.keyvault_name = None
    docker_config = AzureMLDockerConfig(
        base_image="base_image",
        conda_file_path="conda_file_path",
    )

    # assert
    with pytest.raises(ValueError):  # noqa: PT011
        AzureMLSystem(mock_azureml_client_config, "dummy", docker_config, hf_token=True)


@patch("olive.systems.azureml.aml_system.retry_func")
def test__get_enironment_from_config(mock_retry_func):
    # setup
    aml_environment_config = AzureMLEnvironmentConfig(
        name="name",
        version="version",
        label="label",
    )
    mock_azureml_client_config = Mock(spec=AzureMLClientConfig)
    ml_client = MagicMock()
    mock_azureml_client_config.create_client.return_value = ml_client
    mock_azureml_client_config.max_operation_retries = 3
    mock_azureml_client_config.operation_retry_interval = 5
    expected_env = MagicMock()
    mock_retry_func.return_value = expected_env

    # execute
    system = AzureMLSystem(mock_azureml_client_config, "dummy", aml_environment_config=aml_environment_config)

    # assert
    assert expected_env == system.environment


@patch.object(AzureMLClientConfig, "create_client")
def test_create_managed_env(create_client_mock):
    from olive.systems.system_config import AzureMLTargetUserConfig, SystemConfig
    from olive.systems.utils import create_managed_system

    ml_client = MagicMock()
    ml_client.environments.get.side_effect = ResourceNotFoundError()
    ml_client.environments.get.__name__ = "get"
    create_client_mock.return_value = ml_client

    docker_config = AzureMLDockerConfig(
        base_image="base_image",
        conda_file_path="conda_file_path",
    )
    system_config = SystemConfig(
        type="AzureML",
        config=AzureMLTargetUserConfig(
            azureml_client_config=AzureMLClientConfig(),
            aml_compute="aml_compute",
            aml_docker_config=docker_config,
            olive_managed_env=True,
        ),
    )
    system = create_managed_system(system_config, DEFAULT_CPU_ACCELERATOR)
    assert system.config.olive_managed_env

    host_system = create_managed_system(system_config, None)
    assert host_system.config.olive_managed_env

    ml_client = MagicMock()
    ml_client.environments.get.return_value = MagicMock()
    ml_client.environments.get.__name__ = "get"
    create_client_mock.return_value = ml_client

    system_config = SystemConfig(
        type="AzureML",
        config=AzureMLTargetUserConfig(
            azureml_client_config=AzureMLClientConfig(),
            aml_compute="aml_compute",
            aml_environment_config=AzureMLEnvironmentConfig(
                name="name",
                version="version",
                label="label",
            ),
            olive_managed_env=False,
        ),
    )
    system = system_config.create_system()
    assert not system.config.olive_managed_env
