# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
import subprocess
import venv
from pathlib import Path

import pytest


class DependencySetupEnvBuilder(venv.EnvBuilder):
    def post_setup(self, context) -> None:
        super().post_setup(context)
        # Install Olive only
        olive_root = str(Path(__file__).parents[3].resolve())
        subprocess.check_output([context.env_exe, "-Im", "pip", "install", olive_root], stderr=subprocess.STDOUT)


def test_dependency_setup(tmp_path):
    builder = DependencySetupEnvBuilder(with_pip=True)
    builder.create(str(tmp_path))

    if platform.system() == "Windows":
        python_path = tmp_path / "Scripts" / "python"
    else:
        python_path = tmp_path / "bin" / "python"

    user_script_config_file = Path(__file__).parent / "mock_data" / "user_script.json"
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
    else:
        print(result.stdout.decode())

    outputs = subprocess.check_output([python_path, "-Im", "pip", "list"])
    outputs = outputs.decode()
    print(outputs)
    assert "onnxruntime-directml" in outputs
