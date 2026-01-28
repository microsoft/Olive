# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import zipfile
from pathlib import Path
from unittest.mock import patch

import mlflow
import onnx
import pytest

from olive.engine import Engine
from olive.engine.footprint import Footprint, FootprintNode
from olive.engine.output import WorkflowOutput
from olive.engine.packaging.packaging_config import (
    PackagingConfig,
    PackagingType,
)
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.conversion import OnnxConversion
from test.utils import get_accuracy_metric, get_pytorch_model_config


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
        "cache_config": {
            "cache_dir": tmp_path,
            "clean_cache": True,
            "clean_evaluation_cache": True,
        },
        "search_strategy": {
            "execution_order": "joint",
            "sampler": "random",
        },
        "evaluator": evaluator_config,
    }
    engine = Engine(**options)
    # Use TorchScript because dynamo export creates models with strict input shape requirements
    # that don't match the dummy data used for evaluation
    engine.register(OnnxConversion, {"save_as_external_data": save_as_external_data, "use_dynamo_exporter": False})

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_spec=DEFAULT_CPU_ACCELERATOR,
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
    verify_models_rank_json_file(output_dir, models_rank_path, save_as_external_data=save_as_external_data)

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
        "cache_config": {
            "cache_dir": tmp_path,
            "clean_cache": True,
            "clean_evaluation_cache": True,
        },
    }
    engine = Engine(**options)
    engine.register(OnnxConversion, {"use_dynamo_exporter": True})

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_spec=DEFAULT_CPU_ACCELERATOR,
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
    verify_models_rank_json_file(output_dir, models_rank_path)

    # clean up
    shutil.rmtree(output_dir)


# TODO(team): no engine API envolved, use generate_output_artifacts API directly
def test_generate_zipfile_artifacts_mlflow(tmp_path):
    # setup
    options = {
        "cache_config": {
            "cache_dir": tmp_path,
            "clean_cache": True,
            "clean_evaluation_cache": True,
        },
    }
    engine = Engine(**options)
    engine.register(OnnxConversion, {"use_dynamo_exporter": True})

    input_model_config = get_pytorch_model_config()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"
    packaging_config.config.export_in_mlflow_format = True

    output_dir = tmp_path / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        accelerator_spec=DEFAULT_CPU_ACCELERATOR,
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
    verify_models_rank_json_file(output_dir, models_rank_path, export_in_mlflow_format=True)
    assert (output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1" / "mlflow_model").exists()

    # clean up
    shutil.rmtree(output_dir)
    if Path("mlruns").exists():
        shutil.rmtree("mlruns")


def test_generate_zipfile_artifacts_no_output_models(tmp_path):
    # setup
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    model_id = "model_id"
    model_path = "fake_model_file"
    footprint = get_footprint(model_id, model_path)
    footprint.output_model_ids = []  # No output models
    output_dir = tmp_path / "outputs"
    workflow_output = WorkflowOutput(DEFAULT_CPU_ACCELERATOR, footprint)

    # execute
    generate_output_artifacts(packaging_config, workflow_output, output_dir)

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert not artifacts_path.exists()


def test__package_dockerfile(tmp_path):
    # setup
    model_id = "model_id"
    model_path = "fake_model_file"
    footprint = get_footprint(model_id, model_path)
    output_dir = tmp_path / "outputs"

    packaging_config = PackagingConfig(type=PackagingType.Dockerfile)
    workflow_output = WorkflowOutput(DEFAULT_CPU_ACCELERATOR, footprint)

    # execute
    generate_output_artifacts(packaging_config, workflow_output, output_dir)

    # assert
    dockerfile_path = output_dir / "Dockerfile"
    assert dockerfile_path.exists()


def get_footprint(model_id, model_path):
    model_config = {"config": {"model_path": model_path}, "type": "ONNXModel"}
    footprint_node = FootprintNode(model_id=model_id, is_pareto_frontier=True, model_config=model_config)
    footprint = Footprint(nodes={model_id: footprint_node}, is_marked_pareto_frontier=True)
    footprint.input_model_id = model_id
    footprint.output_model_ids = [model_id]  # Mark this as output model
    return footprint


def verify_output_artifacts(output_dir):
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "models_rank.json").exists()


def verify_models_rank_json_file(output_dir, file_path, save_as_external_data=False, export_in_mlflow_format=False):
    with Path.open(file_path) as file:
        data = json.load(file)

    assert data is not None
    # verify model path
    for model_data in data:
        model_path = output_dir / Path(model_data["model_config"]["config"]["model_path"])
        assert model_path.exists(), "Model path in model rank file does not exist."
        if export_in_mlflow_format:
            assert mlflow.onnx.load_model(str(model_path)), (
                "Model path in model rank file is not a valid MLflow model path."
            )
        elif save_as_external_data:
            assert onnx.load(str(model_path / "model.onnx")), (
                "With external data, model path in model rank file is not a valid ONNX model path."
            )
        else:
            assert onnx.load(str(model_path)), "Model path in model rank file is not a valid ONNX model path."
