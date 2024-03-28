# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import zipfile
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_pytorch_model_config
from unittest.mock import Mock, patch

import onnx
import pytest

from olive.engine import Engine
from olive.engine.footprint import Footprint, FootprintNode
from olive.engine.packaging.packaging_config import (
    AzureMLDataPackagingConfig,
    AzureMLModelsPackagingConfig,
    PackagingConfig,
    PackagingType,
)
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx.conversion import OnnxConversion


# TODO(team): no engine API envolved, use generate_output_artifacts API directly
@patch("onnx.external_data_helper.sys.getsizeof")
@pytest.mark.parametrize(
    ("save_as_external_data", "mocked_size_value"),
    [(True, 2048), (False, 100)],
)
def test_generate_zipfile_artifacts(mock_sys_getsizeof, save_as_external_data, mocked_size_value, tmp_path):
    # setup
    # onnx will save with external data when tensor size is greater than 1024(default threshold)
    mock_sys_getsizeof.return_value = mocked_size_value
    metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
    evaluator_config = OliveEvaluatorConfig(metrics=[metric])
    options = {
        "cache_dir": "./cache",
        "clean_cache": True,
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "random",
        },
        "clean_evaluation_cache": True,
    }
    engine = Engine(options, evaluator_config=evaluator_config)
    engine.register(OnnxConversion, {"save_as_external_data": save_as_external_data})

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_specs=[DEFAULT_CPU_ACCELERATOR],
        data_root=None,
        packaging_config=packaging_config,
        output_dir=output_dir,
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    verify_output_artifacts(output_dir)
    models_rank_path = output_dir / "models_rank.json"
    verify_models_rank_json_file(models_rank_path)

    # contain the evaluation result
    candidate_model_path = output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1"
    if save_as_external_data:
        assert (candidate_model_path / "model.onnx.data").exists()
    try:
        model_path = candidate_model_path / "model.onnx"
        onnx.load(str(model_path))
    except Exception as e:
        pytest.fail(f"Failed to load the model: {e}")

    metrics_file = candidate_model_path / "metrics.json"
    with metrics_file.open() as f:
        metrics = json.load(f)
        assert "input_model_metrics" in metrics
        assert "candidate_model_metrics" in metrics

    # clean up
    shutil.rmtree(output_dir)


# TODO(team): no engine API envolved, use generate_output_artifacts API directly
def test_generate_zipfile_artifacts_no_search(tmp_path):
    # setup
    options = {
        "cache_dir": "./cache",
        "clean_cache": True,
        "clean_evaluation_cache": True,
    }
    engine = Engine(options)
    engine.register(OnnxConversion)

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_specs=[DEFAULT_CPU_ACCELERATOR],
        packaging_config=packaging_config,
        output_dir=output_dir,
        evaluate_input_model=False,
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    verify_output_artifacts(output_dir)
    models_rank_path = output_dir / "models_rank.json"
    verify_models_rank_json_file(models_rank_path)

    # clean up
    shutil.rmtree(output_dir)


# TODO(team): no engine API envolved, use generate_output_artifacts API directly
def test_generate_zipfile_artifacts_mlflow(tmp_path):
    # setup
    options = {
        "cache_dir": "./cache",
        "clean_cache": True,
        "clean_evaluation_cache": True,
    }
    engine = Engine(options)
    engine.register(OnnxConversion)

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"
    packaging_config.config.export_in_mlflow_format = True

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_specs=[DEFAULT_CPU_ACCELERATOR],
        packaging_config=packaging_config,
        output_dir=output_dir,
        evaluate_input_model=False,
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    verify_output_artifacts(output_dir)
    models_rank_path = output_dir / "models_rank.json"
    verify_models_rank_json_file(models_rank_path)
    assert (output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1" / "mlflow_model").exists()

    # clean up
    shutil.rmtree(output_dir)


def test_generate_zipfile_artifacts_none_nodes(tmp_path):
    # setup
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    foot_print = Footprint()
    pf_footprint = Footprint()
    pf_footprint.nodes = None
    output_dir = tmp_path / "outputs"

    # execute
    generate_output_artifacts(
        packaging_config, {DEFAULT_CPU_ACCELERATOR: foot_print}, {DEFAULT_CPU_ACCELERATOR: pf_footprint}, output_dir
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert not artifacts_path.exists()


def test_generate_zipfile_artifacts_zero_len_nodes(tmp_path):
    # setup
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    foot_print = Footprint()
    pf_footprint = Footprint()
    pf_footprint.nodes = {}
    output_dir = tmp_path / "outputs"

    # execute
    generate_output_artifacts(
        packaging_config, {DEFAULT_CPU_ACCELERATOR: foot_print}, {DEFAULT_CPU_ACCELERATOR: pf_footprint}, output_dir
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert not artifacts_path.exists()


@patch("olive.engine.packaging.packaging_generator.retry_func")
@patch("olive.engine.packaging.packaging_generator.create_resource_path")
def test_generate_azureml_models(mock_create_resource_path, mock_retry_func):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Model
    from azure.core.exceptions import ServiceResponseError

    version = "1.0"
    description = "Test description"
    name = "OutputModels"
    model_id = "model_id"

    packaging_config = PackagingConfig(
        type=PackagingType.AzureMLModels,
        config=AzureMLModelsPackagingConfig(version=version, description=description),
        name=name,
    )

    model_path = "fake_model_file"

    footprints = get_footprints(model_id, model_path)

    azureml_client_config = Mock(max_operation_retries=3, operation_retry_interval=5)
    ml_client_mock = Mock()
    azureml_client_config.create_client.return_value = ml_client_mock
    resource_path_mock = Mock()
    mock_create_resource_path.return_value = resource_path_mock

    model = Model(
        path=model_path,
        type=AssetTypes.CUSTOM_MODEL,
        name=name,
        version=version,
        description=description,
    )

    # execute
    generate_output_artifacts(
        packaging_config, footprints, footprints, output_dir=Path("output"), azureml_client_config=azureml_client_config
    )

    # assert
    assert mock_retry_func.call_once_with(
        ml_client_mock.models.create_client,
        [model],
        max_tries=azureml_client_config.max_operation_retries,
        delay=azureml_client_config.operation_retry_interval,
        exceptions=ServiceResponseError,
    )


@patch("olive.engine.packaging.packaging_generator.retry_func")
@patch("olive.engine.packaging.packaging_generator.create_resource_path")
def test_generate_azureml_data(mock_create_resource_path, mock_retry_func):
    from azure.ai.ml.constants import AssetTypes
    from azure.ai.ml.entities import Data
    from azure.core.exceptions import ServiceResponseError

    version = "1.0"
    description = "Test description"
    name = "OutputModels"
    model_id = "model_id"

    packaging_config = PackagingConfig(
        type=PackagingType.AzureMLData,
        config=AzureMLDataPackagingConfig(version=version, description=description),
        name=name,
    )

    model_path = "fake_model_file"

    footprints = get_footprints(model_id, model_path)

    azureml_client_config = Mock(max_operation_retries=3, operation_retry_interval=5)
    ml_client_mock = Mock()
    azureml_client_config.create_client.return_value = ml_client_mock
    resource_path_mock = Mock()
    mock_create_resource_path.return_value = resource_path_mock

    data = Data(
        path=model_path,
        type=AssetTypes.URI_FILE,
        name=name,
        version=version,
        description=description,
    )

    # execute
    generate_output_artifacts(
        packaging_config, footprints, footprints, output_dir=Path("output"), azureml_client_config=azureml_client_config
    )

    # assert
    assert mock_retry_func.call_once_with(
        ml_client_mock.models.create_client,
        [data],
        max_tries=azureml_client_config.max_operation_retries,
        delay=azureml_client_config.operation_retry_interval,
        exceptions=ServiceResponseError,
    )


def get_footprints(model_id, model_path):
    acc_spec = AcceleratorSpec(accelerator_type="cpu", execution_provider="CPUExecutionProvider")
    model_config = {"config": {"model_path": model_path}, "type": "ONNXModel"}
    footprint_node = FootprintNode(model_id=model_id, is_pareto_frontier=True, model_config=model_config)
    footprint = Footprint(nodes={model_id: footprint_node}, is_marked_pareto_frontier=True)
    return {acc_spec: footprint}


def verify_output_artifacts(output_dir):
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "models_rank.json").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()


def verify_models_rank_json_file(file_path):
    with Path.open(file_path) as file:
        data = json.load(file)
    assert data is not None
