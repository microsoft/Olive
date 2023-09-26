# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import tempfile
import zipfile
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_pytorch_model_config
from unittest.mock import patch

import onnx
import pytest

from olive.engine import Engine
from olive.engine.footprint import Footprint
from olive.engine.packaging.packaging_config import PackagingConfig, PackagingType
from olive.engine.packaging.packaging_generator import generate_output_artifacts
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.passes.onnx.conversion import OnnxConversion


@patch("onnx.external_data_helper.sys.getsizeof")
@pytest.mark.parametrize(
    "save_as_external_data, mocked_size_value",
    [(True, 2048), (False, 100)],
)
def test_generate_zipfile_artifacts(mock_sys_getsizeof, save_as_external_data, mocked_size_value):
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

    tempdir = tempfile.TemporaryDirectory()
    output_dir = Path(tempdir.name) / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config, data_root=None, packaging_config=packaging_config, output_dir=output_dir
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()

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


def test_generate_zipfile_artifacts_no_search():
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

    tempdir = tempfile.TemporaryDirectory()
    output_dir = Path(tempdir.name) / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        packaging_config=packaging_config,
        output_dir=output_dir,
        evaluate_input_model=False,
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()


def test_generate_zipfile_artifacts_mlflow():
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
    packaging_config.export_in_mlflow_format = True

    tempdir = tempfile.TemporaryDirectory()
    output_dir = Path(tempdir.name) / "outputs"

    # execute
    engine.run(
        input_model_config=input_model_config,
        packaging_config=packaging_config,
        output_dir=output_dir,
        evaluate_input_model=False,
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path) as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()
    assert (output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1" / "mlflow_model").exists()


def test_generate_zipfile_artifacts_none_nodes():
    # setup
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    foot_print = Footprint()
    pf_footprint = Footprint()
    pf_footprint.nodes = None
    tempdir = tempfile.TemporaryDirectory()
    output_dir = Path(tempdir.name) / "outputs"

    # execute
    generate_output_artifacts(
        packaging_config, {DEFAULT_CPU_ACCELERATOR: foot_print}, {DEFAULT_CPU_ACCELERATOR: pf_footprint}, output_dir
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert not artifacts_path.exists()


def test_generate_zipfile_artifacts_zero_len_nodes():
    # setup
    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    foot_print = Footprint()
    pf_footprint = Footprint()
    pf_footprint.nodes = {}
    tempdir = tempfile.TemporaryDirectory()
    output_dir = Path(tempdir.name) / "outputs"

    # execute
    generate_output_artifacts(
        packaging_config, {DEFAULT_CPU_ACCELERATOR: foot_print}, {DEFAULT_CPU_ACCELERATOR: pf_footprint}, output_dir
    )

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert not artifacts_path.exists()
