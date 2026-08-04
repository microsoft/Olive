# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.common.constants import OS
from olive.workflows.run.run import run as olive_run

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


def test_dependency_setup(config_json):
    with patch("olive.workflows.run.run.generate_files_from_packages") as mock_generate:
        olive_run(config_json, list_required_packages=True)

    required_packages, output_path = mock_generate.call_args.args

    ort_extra = "onnxruntime-directml" if platform.system() == OS.WINDOWS else "onnxruntime-gpu"

    assert output_path == "olive_requirements.txt"
    assert ort_extra in required_packages
    assert "psutil" in required_packages
