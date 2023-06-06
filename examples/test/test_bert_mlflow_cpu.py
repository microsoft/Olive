# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
from pathlib import Path

import pytest
from utils import check_search_output


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


# Skip docker_system test until bug is fixed: https://github.com/docker/docker-py/issues/3113
@pytest.mark.parametrize("olive_json", ["bert_mlflow_cpu.json"])
def test_bert(olive_json):

    from olive.workflows import run as olive_run

    with open(olive_json, "r") as fin:
        olive_config = json.load(fin)

    if olive_json == "bert_mlflow_cpu.json":
        output_dir = Path(__file__).parent / "outputs"
        olive_config["engine"]["output_dir"] = output_dir

    footprint = olive_run(olive_config)
    check_search_output(footprint)

    if olive_json == "bert_mlflow_cpu.json":
        artifacts_path = output_dir / "OutputModels.zip"
        check_mlflow_output(artifacts_path, output_dir)


def check_mlflow_output(artifacts_path, output_dir):
    import zipfile

    import mlflow

    assert artifacts_path.exists()
    with zipfile.ZipFile(artifacts_path, "r") as zip_ref:
        zip_ref.extractall(output_dir)
    assert (output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1" / "model" / "MLmodel").exists()
    assert mlflow.pyfunc.load_model(output_dir / "CandidateModels" / "cpu-cpu" / "BestCandidateModel_1" / "model")
