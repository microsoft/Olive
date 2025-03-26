# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os

import pytest
from onnxruntime import __version__ as OrtVersion
from packaging import version

from olive.common.utils import retry_func, run_subprocess

from ..utils import check_output, get_example_dir, patch_config


@pytest.fixture(scope="module", autouse=True)
def setup():
    """Setups any state specific to the execution of the given module."""
    os.chdir(get_example_dir("resnet"))

    # prepare model and data
    # retry since it fails randomly
    retry_func(run_subprocess, kwargs={"cmd": "python prepare_model_data.py", "check": True})


@pytest.mark.parametrize("sampler", ["random"])
@pytest.mark.parametrize("execution_order", ["pass-by-pass"])
@pytest.mark.parametrize("system", ["local_system"])
@pytest.mark.parametrize(
    "olive_json",
    [
        "resnet_ptq_cpu.json",
        # TODO(trajep): remove skip once the bug of azureml-fsspec is fixed
        pytest.param(
            "resnet_ptq_cpu_aml_dataset.json", marks=pytest.mark.skip(reason="credential bug in azureml-fsspec")
        ),
    ],
)
@pytest.mark.skipif(
    version.parse(OrtVersion) == version.parse("1.16.0"),
    reason="resnet is not supported in ORT 1.16.0 caused by https://github.com/microsoft/onnxruntime/issues/17627",
)
def test_resnet(sampler, execution_order, system, olive_json):
    from olive.workflows import run as olive_run

    olive_config = patch_config(olive_json, sampler, execution_order, system)

    footprint = olive_run(olive_config, tempdir=os.environ.get("OLIVE_TEMPDIR", None))
    check_output(footprint)
