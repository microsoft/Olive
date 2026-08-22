# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import shutil
import zipfile
from pathlib import Path

import onnx
import pytest

from olive.engine.footprint import Footprint, FootprintNode, FootprintNodeMetric
from olive.engine.output import WorkflowOutput
from olive.engine.packaging.packaging_config import (
    PackagingConfig,
    PackagingType,
    ZipfilePackagingConfig,
)
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric_result import MetricResult, SubMetricResult
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from test.utils import ONNX_MODEL_PATH


@pytest.mark.parametrize(
    "save_as_external_data",
    [
        pytest.param(False, id="inline-onnx-with-metrics"),
        pytest.param(True, id="external-data-onnx-with-metrics"),
    ],
)
def test_generate_zipfile_artifacts_with_metrics(tmp_path, save_as_external_data):
    packaging_config = PackagingConfig(type=PackagingType.Zipfile, name="OutputModels")
    workflow_output = create_workflow_output(
        create_test_model_config(tmp_path, save_as_external_data=save_as_external_data),
        include_metrics=True,
    )
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    generate_output_artifacts(packaging_config, workflow_output, output_dir)

    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)

    verify_output_artifacts(output_dir)
    models_rank_path = output_dir / "models_rank.json"
    verify_models_rank_json_file(output_dir, models_rank_path, save_as_external_data=save_as_external_data)

    candidate_model_path = output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1"
    if save_as_external_data:
        assert (candidate_model_path / "model.onnx.data").exists()
    assert_onnx_loads(candidate_model_path / "model.onnx")

    metrics_file = candidate_model_path / "metrics.json"
    with metrics_file.open() as f:
        metrics = json.load(f)
    assert "input_model_metrics" in metrics
    assert "candidate_model_metrics" in metrics


@pytest.mark.parametrize(
    "export_in_mlflow_format",
    [
        pytest.param(False, id="zip-without-metrics-or-search"),
        pytest.param(True, id="mlflow-without-metrics"),
    ],
)
def test_generate_zipfile_artifacts_without_metrics(tmp_path, export_in_mlflow_format):
    packaging_config = PackagingConfig(
        type=PackagingType.Zipfile,
        name="OutputModels",
        config=ZipfilePackagingConfig(export_in_mlflow_format=export_in_mlflow_format),
    )
    workflow_output = create_workflow_output(create_test_model_config(tmp_path), include_metrics=False)
    output_dir = tmp_path / "outputs"
    output_dir.mkdir()

    generate_output_artifacts(packaging_config, workflow_output, output_dir)

    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)

    verify_output_artifacts(output_dir)
    models_rank_path = output_dir / "models_rank.json"
    verify_models_rank_json_file(
        output_dir,
        models_rank_path,
        export_in_mlflow_format=export_in_mlflow_format,
    )

    candidate_model_path = output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1"
    assert not (candidate_model_path / "metrics.json").exists()
    if export_in_mlflow_format:
        assert (candidate_model_path / "mlflow_model").exists()
        if Path("mlruns").exists():
            shutil.rmtree("mlruns")
    else:
        assert_onnx_loads(candidate_model_path / "model.onnx")


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


def create_test_model_config(tmp_path, save_as_external_data=False):
    if not save_as_external_data:
        return {"type": "ONNXModel", "config": {"model_path": str(ONNX_MODEL_PATH)}}

    model_proto = onnx.load(ONNX_MODEL_PATH)
    model_dir = tmp_path / "external_data_model"
    model_dir.mkdir()
    onnx.save_model(
        model_proto,
        model_dir / "model.onnx",
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location="model.onnx.data",
        size_threshold=0,
    )
    return {
        "type": "ONNXModel",
        "config": {"model_path": str(model_dir), "onnx_file_name": "model.onnx"},
    }


def create_workflow_output(model_config, include_metrics):
    input_model_id = "input_model"
    output_model_id = "candidate_model"
    input_node = FootprintNode(
        model_id=input_model_id,
        model_config={"type": "ONNXModel", "config": {"model_path": str(ONNX_MODEL_PATH)}},
        metrics=create_accuracy_metrics(0.80) if include_metrics else None,
        pass_run_config={"type": "input_model"},
    )
    output_node = FootprintNode(
        model_id=output_model_id,
        parent_model_id=input_model_id,
        model_config=model_config,
        metrics=create_accuracy_metrics(0.90) if include_metrics else None,
        from_pass="test_pass",
        pass_run_config={"type": "test_pass"},
        is_pareto_frontier=True,
    )
    objective_dict = {"accuracy": {"higher_is_better": True, "goal": None, "priority": 1}} if include_metrics else {}
    footprint = Footprint(
        nodes={input_model_id: input_node, output_model_id: output_node},
        objective_dict=objective_dict,
        is_marked_pareto_frontier=True,
    )
    footprint.input_model_id = input_model_id
    footprint.output_model_ids = [output_model_id]
    return WorkflowOutput(DEFAULT_CPU_ACCELERATOR, footprint)


def create_accuracy_metrics(value):
    return FootprintNodeMetric(
        value=MetricResult(root={"accuracy": SubMetricResult(value=value, priority=1, higher_is_better=True)}),
        cmp_direction={"accuracy": 1},
    )


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


def assert_onnx_loads(model_path):
    try:
        onnx.load(str(model_path))
    except Exception as e:
        pytest.fail(f"Failed to load the model: {e}")


def verify_models_rank_json_file(output_dir, file_path, save_as_external_data=False, export_in_mlflow_format=False):
    with file_path.open() as file:
        data = json.load(file)

    assert data is not None
    # verify model path
    for model_data in data:
        model_path = output_dir / Path(model_data["model_config"]["config"]["model_path"])
        assert model_path.exists(), "Model path in model rank file does not exist."
        if export_in_mlflow_format:
            import mlflow

            assert mlflow.onnx.load_model(str(model_path)), (
                "Model path in model rank file is not a valid MLflow model path."
            )
        elif save_as_external_data:
            assert_onnx_loads(model_path / "model.onnx")
        else:
            assert_onnx_loads(model_path)
