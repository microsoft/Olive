# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import shutil
import tempfile
from pathlib import Path
from test.unit_test.utils import ONNX_MODEL_PATH, get_accuracy_metric, get_latency_metric, get_pytorch_model
from unittest.mock import MagicMock, Mock, patch

import pytest
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricResult
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ModelStorageKind, ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
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
        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client

        # execute
        res = self.system.evaluate_model(olive_model, [metric], DEFAULT_CPU_ACCELERATOR)

        # assert
        mock_create_pipeline.assert_called_once_with(output_folder, olive_model, [metric], DEFAULT_CPU_ACCELERATOR)
        ml_client.jobs.stream.assert_called_once()
        assert mock_retry_func.call_count == 2
        if metric.name == "accuracy":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.99618
        if metric.name == "latency":
            for sub_type in metric.sub_types:
                assert res.get_value(metric.name, sub_type.name) == 0.031415

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
        ml_client = MagicMock()
        self.system.azureml_client_config.create_client.return_value = ml_client

        # execute
        actual_res = self.system.run_pass(p, olive_model, output_model_path)

        # assert
        mock_create_pipeline.assert_called_once_with(output_folder, olive_model, p.to_json(), p.path_params)
        assert mock_retry_func.call_count == 2
        ml_client.jobs.stream.assert_called_once()
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
        sub_type_name = ",".join([st.name for st in metric.sub_types])
        metric_type = f"{metric.type}-{sub_type_name}"
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
        accelerator_config_path = tem_dir / "accelerator_config.json"
        inputs = {
            **model_inputs,
            **metric_inputs,
            "accelerator_config": Input(type=AssetTypes.URI_FILE),
        }
        expected_res = MagicMock()
        mock_command.return_value.return_value = expected_res

        # execute
        actual_res = self.system._create_metric_component(
            tem_dir, metric, model_args, model_storage_kind, accelerator_config_path
        )

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
                    "use_ort_implementation": False,
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
                "model_storage_kind": "folder",
                "name": None,
                "script_dir": None,
                "version": None,
            },
            "type": "PyTorchModel",
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
                    "use_ort_implementation": False,
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
                "model_storage_kind": "folder",
                "name": None,
                "script_dir": None,
                "version": None,
            },
            "type": "PyTorchModel",
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
