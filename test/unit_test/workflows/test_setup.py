# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import shutil
import venv
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.common.utils import run_subprocess

# pylint: disable=redefined-outer-name


class DependencySetupEnvBuilder(venv.EnvBuilder):
    def post_setup(self, context) -> None:
        super().post_setup(context)
        # Install Olive only
        olive_root = str(Path(__file__).parents[3].resolve())
        run_subprocess([context.env_exe, "-Im", "pip", "install", olive_root], check=True)


@pytest.fixture()
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
    builder = DependencySetupEnvBuilder(with_pip=True)
    builder.create(str(tmp_path))

    if platform.system() == OS.WINDOWS:
        python_path = tmp_path / "Scripts" / "python"
        ort_extra = "onnxruntime-directml"
    else:
        python_path = tmp_path / "bin" / "python"
        ort_extra = "onnxruntime-gpu"

    user_script_config_file = config_json
    cmd = [
        python_path,
        "-Im",
        "olive.workflows.run",
        "--config",
        str(user_script_config_file),
        "--setup",
    ]

    return_code, _, stderr = run_subprocess(cmd, check=True)
    if return_code != 0:
        pytest.fail(stderr)

    _, outputs, _ = run_subprocess([python_path, "-Im", "pip", "list"], check=True)
    assert ort_extra in outputs
    assert "psutil" in outputs
    shutil.rmtree(tmp_path, ignore_errors=True)
