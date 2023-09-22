# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version
from utils import check_output, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """setup any state specific to the execution of the given module."""
    cur_dir = Path(__file__).resolve().parent.parent
    example_dir = cur_dir / "bert"
    os.chdir(example_dir)
    yield
    os.chdir(cur_dir)


@pytest.mark.parametrize("search_algorithm", ["tpe"])
@pytest.mark.parametrize("execution_order", ["joint", "pass-by-pass"])
@pytest.mark.parametrize("system", ["aml_system"])
@pytest.mark.parametrize("olive_json", ["bert_cuda_gpu.json"])
@pytest.mark.parametrize("enable_cuda_graph", [True, False])
@pytest.mark.skipif(
    version.parse(OrtVersion) == version.parse("1.16.0"),
    reason=(
        "Quantization is not supported in ORT 1.16.0,"
        " caused by https://github.com/microsoft/onnxruntime/issues/17619"
    ),
)
def test_bert(search_algorithm, execution_order, system, olive_json, enable_cuda_graph):

    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, search_algorithm, execution_order, system, is_gpu=True)
    olive_config["passes"]["perf_tuning"]["config"]["enable_cuda_graph"] = enable_cuda_graph

    footprint = olive_run(olive_config)
    check_output(footprint)
