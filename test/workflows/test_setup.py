# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import shutil
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.common.utils import run_subprocess

# pylint: disable=redefined-outer-name


@pytest.fixture
def config_json(tmp_path):
    if platform.system() == OS.WINDOWS:
        ep = "DmlExecutionProvider"
    else:
        ep = "CUDAExecutionProvider"

    with (Path(__file__).parent / "mock_data" / "dependency_setup.json").open() as f:
        config = json.load(f)
        config["systems"]["local_system"]["accelerators"][0]["execution_providers"] = [ep]

    config_json_file = tmp_path / "config.json"
    with config_json_file.open("w") as f:
        json.dump(config, f)

    return str(config_json_file)


def test_dependency_setup(tmp_path, config_json):
    cmd = [
        "olive",
        "run",
        "--config",
        str(config_json),
        "--list_required_packages",
    ]

    return_code, _, stderr = run_subprocess(cmd, check=False)
    if return_code != 0:
        pytest.fail(stderr)

    output_filepath = Path("olive_requirements.txt")
    assert output_filepath.exists()

    required_packages = []
    with output_filepath.open() as strm:
        required_packages = [_.strip() for _ in strm.readlines()]

    ort_extra = "onnxruntime-directml" if platform.system() == OS.WINDOWS else "onnxruntime-gpu"

    assert ort_extra in required_packages
    assert "psutil" in required_packages
    output_filepath.unlink()
    shutil.rmtree(tmp_path, ignore_errors=True)
