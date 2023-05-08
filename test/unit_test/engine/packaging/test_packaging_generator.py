# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
import zipfile
from pathlib import Path
from test.unit_test.utils import get_accuracy_metric, get_pytorch_model

from olive.engine import Engine
from olive.engine.packaging.packaging_config import PackagingConfig, PackagingType
from olive.evaluator.metric import AccuracySubType
from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.passes.onnx.conversion import OnnxConversion


def test_generate_zipfile_artifacts():
    # setup
    metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE)
    evaluator = OliveEvaluator(metrics=[metric])
    options = {
        "cache_dir": "./cache",
        "clean_cache": True,
        "search_strategy": {
            "execution_order": "joint",
            "search_algorithm": "random",
        },
        "clean_evaluation_cache": True,
    }
    engine = Engine(options, evaluator=evaluator)
    engine.register(OnnxConversion, {}, False)

    input_model = get_pytorch_model()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = Path(__file__).parent / "outputs"

    # execute
    engine.run(input_model=input_model, packaging_config=packaging_config, output_dir=output_dir)

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()

    # cleanup
    shutil.rmtree(output_dir)


def test_generate_zipfile_artifacts_no_search():
    # setup
    options = {
        "cache_dir": "./cache",
        "clean_cache": True,
        "clean_evaluation_cache": True,
    }
    engine = Engine(options)
    engine.register(OnnxConversion, {}, False)

    input_model = get_pytorch_model()

    packaging_config = PackagingConfig()
    packaging_config.type = PackagingType.Zipfile
    packaging_config.name = "OutputModels"

    output_dir = Path(__file__).parent / "outputs"

    # execute
    engine.run(input_model=input_model, packaging_config=packaging_config, output_dir=output_dir)

    # assert
    artifacts_path = output_dir / "OutputModels.zip"
    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "SampleCode").exists()
    assert (output_dir / "CandidateModels").exists()
    assert (output_dir / "ONNXRuntimePackages").exists()

    # cleanup
    shutil.rmtree(output_dir)
