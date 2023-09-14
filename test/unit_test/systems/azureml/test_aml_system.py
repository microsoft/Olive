# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import shutil
import tempfile
from pathlib import Path
from test.unit_test.utils import ONNX_MODEL_PATH, get_accuracy_metric, get_latency_metric, get_pytorch_model_config
from unittest.mock import MagicMock, Mock, patch

import pytest
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricResult
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.resource_path import AzureMLModel, ResourcePath, ResourceType, create_resource_path
from olive.systems.azureml.aml_evaluation_runner import main as aml_evaluation_runner_main
from olive.systems.azureml.aml_pass_runner import main as aml_pass_runner_main
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
        (get_accuracy_metric(AccuracySubType.AUROC)),
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
        model_config = get_pytorch_model_config()
        output_folder = Path(__file__).absolute().parent / "output_metrics"
        mock_tempdir.return_value.__enter__.return_value = output_folder
        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client
        self.system.azureml_client_config.max_operation_retries = 3
        self.system.azureml_client_config.operation_retry_interval = 5

        # execute
        res = self.system.evaluate_model(model_config, None, [metric], DEFAULT_CPU_ACCELERATOR)

        # assert
        mock_create_pipeline.assert_called_once_with(
            None, output_folder, model_config, [metric], DEFAULT_CPU_ACCELERATOR
        )
        ml_client.jobs.stream.assert_called_once()
        assert mock_retry_func.call_count == 2
        if metric.name == "accuracy":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.99618
        if metric.name == "latency":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.031415

    @patch("olive.systems.azureml.aml_system.retry_func")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_pipeline_for_pass")
    def test_run_pass(self, mock_create_pipeline, mock_retry_func):
        # setup
        tmp_dir = tempfile.TemporaryDirectory()
        tmp_dir_path = Path(tmp_dir.name)
        # dummy pipeline output download path
        pipeline_output_path = tmp_dir_path / "pipeline_output" / "named-outputs" / "pipeline_output"
        pipeline_output_path.mkdir(parents=True, exist_ok=True)
        # create dummy output model
        downloaded_output_model_path = pipeline_output_path / "output_model.onnx"
        with open(downloaded_output_model_path, "w") as f:
            f.write("dummy")
        # create dummy output config
        dummy_config = {
            "type": "ONNXModel",
            "config": {
                "model_path": {"type": "file", "config": {"path": "output_model.onnx"}},
                "inference_settings": None,
            },
            "same_resources_as_input": [],
            "resource_names": ["model_path"],
        }
        dummy_config_path = pipeline_output_path / "output_model_config.json"
        with open(dummy_config_path, "w") as f:
            json.dump(dummy_config, f, indent=4)

        onnx_conversion_config = {}
        p = create_pass_from_dict(OnnxConversion, onnx_conversion_config)
        model_config = get_pytorch_model_config()
        output_model_path = tmp_dir_path / "output_folder" / "output_model_path"
        output_model_path.mkdir(parents=True, exist_ok=True)
        # create dummy output model so that ONNXModel can be created with the same path
        expected_model_path = output_model_path / "model.onnx"
        with open(expected_model_path, "w") as f:
            f.write("dummy")
        output_folder = tmp_dir_path

        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client
        self.system.azureml_client_config.max_operation_retries = 3
        self.system.azureml_client_config.operation_retry_interval = 5

        with patch("olive.systems.azureml.aml_system.tempfile.TemporaryDirectory") as mock_tempdir:
            mock_tempdir.return_value.__enter__.return_value = output_folder
            # execute
            actual_res = self.system.run_pass(p, model_config, None, output_model_path)

        # assert
        mock_create_pipeline.assert_called_once_with(None, output_folder, model_config, p.to_json(), p.path_params)
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
    def test__create_model_args(self, model_resource_type):
        # setup
        temp_model = tempfile.NamedTemporaryFile(dir=".", suffix=".onnx", prefix="model_0")
        ws_config = {
            "workspace_name": "workspace_name",
            "subscription_id": "subscription_id",
            "resource_group": "resource_group",
        }
        self.system.azureml_client_config.get_workspace_config.return_value = ws_config
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
            },
        }
        tem_dir = tempfile.TemporaryDirectory()
        tem_dir_path = Path(tem_dir.name)
        model_config_path = tem_dir_path / "model_config.json"
        if model_resource_type == ResourceType.AzureMLModel:
            expected_model_path = Input(type=AssetTypes.CUSTOM_MODEL, path="azureml:model_name:version")
        elif model_resource_type == ResourceType.LocalFile:
            expected_model_path = Input(type=AssetTypes.URI_FILE, path=Path(temp_model.name).resolve())
        else:
            expected_model_path = None
        expected_model_config = Input(type=AssetTypes.URI_FILE, path=model_config_path)
        expected_res = {"model_config": expected_model_config, "model_model_path": expected_model_path}

        # execute
        actual_res = self.system._create_model_args(
            model_json, {"model_path": create_resource_path(model_json["config"]["model_path"])}, tem_dir_path
        )

        # assert
        assert actual_res == expected_res
        if model_resource_type != ResourceType.StringName:
            assert model_json["config"]["model_path"] is None
        else:
            assert model_json["config"]["model_path"] == resource_paths[model_resource_type]

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
        inputs = {
            "dummy_input": Input(type=AssetTypes.URI_FILE),
        }
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
            script_name,
        )

        # assert
        assert actual_res == expected_res
        mock_command.assert_called_once_with(
            name=name,
            display_name=display_name,
            description=description,
            command=self.create_command(inputs),
            resources=resources,
            environment=aml_environment,
            code=code,
            inputs=inputs,
            outputs=dict(pipeline_output=Output(type=AssetTypes.URI_FOLDER)),
            instance_count=1,
            compute=compute,
        )

    def test__create_metric_args(self):
        # setup
        # the reason why we need resolve: sometimes, windows system would change c:\\ to C:\\ when calling resolve.
        tem_dir = Path(__file__).absolute().parent.resolve()
        metric_config = {
            "user_config": {
                "user_script": "user_script",
                "script_dir": "script_dir",
                "data_dir": tem_dir,
            }
        }
        metric_config_path = tem_dir / "metric_config.json"

        expected_metric_config = Input(type=AssetTypes.URI_FILE, path=metric_config_path)
        expected_metric_user_script = Input(type=AssetTypes.URI_FILE, path="user_script")
        expected_metric_script_dir = Input(type=AssetTypes.URI_FOLDER, path="script_dir")
        expected_metric_data_dir = Input(type=AssetTypes.URI_FOLDER, path=str(tem_dir))
        expected_res = {
            "metric_config": expected_metric_config,
            "metric_user_script": expected_metric_user_script,
            "metric_script_dir": expected_metric_script_dir,
            "metric_data_dir": expected_metric_data_dir,
        }

        # execute
        actual_res = self.system._create_metric_args(None, metric_config, tem_dir)

        # assert
        assert actual_res == expected_res
        assert metric_config["user_config"]["user_script"] is None
        assert metric_config["user_config"]["script_dir"] is None
        assert metric_config["user_config"]["data_dir"] is None

        # cleanup
        if os.path.exists(metric_config_path):
            os.remove(metric_config_path)

    @pytest.mark.parametrize(
        "model_resource_type",
        [ResourceType.AzureMLModel, ResourceType.LocalFile],
    )
    @patch("olive.systems.azureml.aml_system.shutil.copy")
    @patch("olive.systems.azureml.aml_system.command")
    @patch("olive.systems.azureml.aml_system.AzureMLSystem._create_metric_args")
    def test__create_metric_component(self, mock_create_metric_args, mock_command, mock_copy, model_resource_type):
        # setup
        tem_dir = Path(".")
        code_path = tem_dir / "code"
        metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
        metric.user_config = {}
        model_args = {"input": Input(type=AssetTypes.URI_FILE, path="path")}
        sub_type_name = ",".join([st.name for st in metric.sub_types])
        metric_type = f"{metric.type}-{sub_type_name}"
        model_inputs = {
            "model_config": Input(type=AssetTypes.URI_FILE),
            "model_model_path": Input(type=AssetTypes.CUSTOM_MODEL, optional=True)
            if model_resource_type == ResourceType.AzureMLModel
            else Input(type=AssetTypes.URI_FILE, optional=True),
        }
        metric_inputs = {
            "metric_config": Input(type=AssetTypes.URI_FILE),
            "metric_user_script": Input(type=AssetTypes.URI_FILE, optional=True),
            "metric_script_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
            "metric_data_dir": Input(type=AssetTypes.URI_FOLDER, optional=True),
        }
        accelerator_config_path = tem_dir / "accelerator_config.json"
        inputs = {
            **model_inputs,
            **metric_inputs,
            "accelerator_config": Input(type=AssetTypes.URI_FILE),
        }
        expected_res = MagicMock()
        mock_command.return_value.return_value = expected_res

        # execute
        model_resource_path = None
        if model_resource_type == ResourceType.AzureMLModel:
            model_resource_path = AzureMLModel(
                {
                    "azureml_client": {
                        "subscription_id": "subscription_id",
                        "resource_group": "resource_group",
                        "workspace_name": "workspace_name",
                    },
                    "name": "test",
                    "version": "1",
                }
            )
        else:
            model_resource_path = create_resource_path(ONNX_MODEL_PATH)
        actual_res = self.system._create_metric_component(
            None, tem_dir, metric, model_args, {"model_path": model_resource_path}, accelerator_config_path
        )

        # assert
        assert actual_res == expected_res
        mock_command.assert_called_once_with(
            name=metric_type,
            display_name=metric_type,
            description=f"Run olive {metric_type} evaluation",
            command=self.create_command(inputs),
            resources=None,
            environment=self.system.environment,
            code=str(code_path),
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

    @patch("olive.evaluator.olive_evaluator.OliveEvaluator.evaluate")
    def test_aml_evaluation_runner(self, mock_evaluate, tmp_path):
        mock_evaluate.return_value = MetricResult.parse_obj(
            {"accuracy-accuracy_score": {"value": 0.5, "priority": 1, "higher_is_better": True}}
        )

        # create model_config.json
        model_config = {
            "config": {
                "dummy_inputs_func": None,
                "hf_config": {
                    "components": None,
                    "dataset": {
                        "batch_size": 1,
                        "data_name": "glue",
                        "input_cols": ["sentence1", "sentence2"],
                        "label_cols": ["label"],
                        "split": "validation",
                        "subset": "mrpc",
                    },
                    "model_class": None,
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                },
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
                    "output_shapes": None,
                    "output_types": None,
                    "string_to_int_dim_params": None,
                },
                "model_loader": None,
                "model_path": None,
                "model_script": None,
                "script_dir": None,
            },
            "type": "PyTorchModel",
            "resource_names": [],
        }

        with open(tmp_path / "model_config.json", "w") as f:
            json.dump(model_config, f)

        # create model.pt
        # create metrics_config.json
        metrics_config = {
            "data_config": {
                "components": {
                    "dataloader": {
                        "name": "default_dataloader",
                        "params": {"batch_size": 1},
                        "type": "default_dataloader",
                    },
                    "load_dataset": {
                        "name": "huggingface_dataset",
                        "params": {"data_name": "glue", "split": "validation", "subset": "mrpc"},
                        "type": "huggingface_dataset",
                    },
                    "post_process_data": {
                        "name": "text_classification_post_process",
                        "params": {},
                        "type": "text_classification_post_process",
                    },
                    "pre_process_data": {
                        "name": "huggingface_pre_process",
                        "params": {
                            "input_cols": ["sentence1", "sentence2"],
                            "label_cols": ["label"],
                            "model_name": "Intel/bert-base-uncased-mrpc",
                        },
                        "type": "huggingface_pre_process",
                    },
                },
                "default_components": {
                    "dataloader": {"name": "default_dataloader", "params": {}, "type": "default_dataloader"},
                    "load_dataset": {"name": "huggingface_dataset", "params": {}, "type": "huggingface_dataset"},
                    "post_process_data": {
                        "name": "text_classification_post_process",
                        "params": {},
                        "type": "text_classification_post_process",
                    },
                    "pre_process_data": {
                        "name": "huggingface_pre_process",
                        "params": {},
                        "type": "huggingface_pre_process",
                    },
                },
                "default_components_type": {
                    "dataloader": "default_dataloader",
                    "load_dataset": "huggingface_dataset",
                    "post_process_data": "text_classification_post_process",
                    "pre_process_data": "huggingface_pre_process",
                },
                "name": "_default_huggingface_dc",
                "params_config": {
                    "batch_size": 1,
                    "data_name": "glue",
                    "input_cols": ["sentence1", "sentence2"],
                    "label_cols": ["label"],
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "split": "validation",
                    "subset": "mrpc",
                    "task_type": "text-classification",
                },
                "type": "HuggingfaceContainer",
            },
            "name": "result",
            "sub_types": [
                {
                    "name": "accuracy_score",
                }
            ],
            "type": "accuracy",
            "user_config": {
                "batch_size": 1,
                "data_dir": None,
                "dataloader_func": None,
                "inference_settings": None,
                "input_names": None,
                "input_shapes": None,
                "input_types": None,
                "post_processing_func": None,
                "script_dir": None,
                "user_script": None,
            },
        }

        with open(tmp_path / "metrics_config.json", "w") as f:
            json.dump(metrics_config, f)

        # create accelerator_config.json
        accelerator_config = {"accelerator_type": "cpu", "execution_provider": "CPUExecutionProvider"}
        with open(tmp_path / "accelerator_config.json", "w") as f:
            json.dump(accelerator_config, f)

        ouptut_dir = tmp_path / "pipeline_output"
        ouptut_dir.mkdir()

        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--metric_config",
            str(tmp_path / "metrics_config.json"),
            "--accelerator_config",
            str(tmp_path / "accelerator_config.json"),
            "--pipeline_output",
            str(ouptut_dir),
        ]
        aml_evaluation_runner_main(args)
        mock_evaluate.assert_called_once()

    @patch("olive.passes.onnx.conversion.OnnxConversion.run")
    def test_pass_runner(self, mock_conversion_run, tmp_path):
        model_config = {
            "config": {
                "dummy_inputs_func": None,
                "hf_config": {
                    "components": None,
                    "dataset": {
                        "batch_size": 1,
                        "data_name": "glue",
                        "input_cols": ["sentence1", "sentence2"],
                        "label_cols": ["label"],
                        "split": "validation",
                        "subset": "mrpc",
                    },
                    "model_class": None,
                    "model_name": "Intel/bert-base-uncased-mrpc",
                    "task": "text-classification",
                },
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
                    "output_shapes": None,
                    "output_types": None,
                    "string_to_int_dim_params": None,
                },
                "model_loader": None,
                "model_path": None,
                "model_script": None,
                "script_dir": None,
            },
            "type": "PyTorchModel",
            "resource_names": [],
        }
        pass_config = {
            "accelerator": {"accelerator_type": "cpu", "execution_provider": "CPUExecutionProvider"},
            "config": {
                "all_tensors_to_one_file": True,
                "external_data_name": None,
                "save_as_external_data": False,
                "script_dir": None,
                "target_opset": 13,
                "user_script": None,
            },
            "disable_search": True,
            "type": "OnnxConversion",
        }

        with open(tmp_path / "model_config.json", "w") as f:
            json.dump(model_config, f)

        with open(tmp_path / "pass_config.json", "w") as f:
            json.dump(pass_config, f)

        ouptut_dir = tmp_path / "pipeline_output"
        ouptut_dir.mkdir()
        shutil.copy(ONNX_MODEL_PATH, ouptut_dir)
        mock_conversion_run.return_value = ONNXModel(ouptut_dir / ONNX_MODEL_PATH.name)

        args = [
            "--model_config",
            str(tmp_path / "model_config.json"),
            "--pass_config",
            str(tmp_path / "pass_config.json"),
            "--pipeline_output",
            str(ouptut_dir),
            "--pass_accelerator_type",
            "cpu",
            "--pass_execution_provider",
            "CPUExecutionProvider",
        ]

        aml_pass_runner_main(args)
        mock_conversion_run.assert_called_once()
