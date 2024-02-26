# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import platform
import subprocess
import venv
from pathlib import Path

import pytest

# pylint: disable=redefined-outer-name


class DependencySetupEnvBuilder(venv.EnvBuilder):
    def post_setup(self, context) -> None:
        super().post_setup(context)
        # Install Olive only
        olive_root = str(Path(__file__).parents[3].resolve())
        subprocess.check_output([context.env_exe, "-Im", "pip", "install", olive_root], stderr=subprocess.STDOUT)


@pytest.fixture()
def config_json(tmp_path):
    if platform.system() == "Windows":
        ep = "DmlExecutionProvider"
    else:
        ep = "CUDAExecutionProvider"

    with (Path(__file__).parent / "mock_data" / "dependency_setup.json").open() as f:
        config = json.load(f)
        config["engine"]["execution_providers"] = [ep]

    config_json_file = tmp_path / "config.json"
    with config_json_file.open("w") as f:
        json.dump(config, f)

    return str(config_json_file)


def test_dependency_setup(tmp_path, config_json):
    builder = DependencySetupEnvBuilder(with_pip=True)
    builder.create(str(tmp_path))

    if platform.system() == "Windows":
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
    # pylint: disable=subprocess-run-check
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)  # noqa: PLW1510

    if result.returncode != 0:
        pytest.fail(result.stdout.decode())

    outputs = subprocess.check_output([python_path, "-Im", "pip", "list"])
    outputs = outputs.decode()
    assert ort_extra in outputs
