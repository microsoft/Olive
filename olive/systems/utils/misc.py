# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import os
import shutil
import tempfile
from copy import deepcopy
from pathlib import Path
from typing import Optional, Union

from olive.common.utils import run_subprocess
from olive.hardware.constants import PROVIDER_PACKAGE_MAPPING

logger = logging.getLogger(__name__)


def get_package_name_from_ep(execution_provider: str) -> str:
    """Get the package name from the execution provider."""
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, "onnxruntime")


def create_new_environ(
    environment_variables: Optional[dict[str, str]] = None,
    prepend_to_path: Optional[list[str]] = None,
    python_environment_path: Optional[Union[Path, str]] = None,
):
    """Create a copy of the current environment with the given environment variables and paths prepended."""
    environ = deepcopy(os.environ)
    if environment_variables:
        environ.update(environment_variables)
    if prepend_to_path:
        environ["PATH"] = os.pathsep.join(prepend_to_path) + os.pathsep + environ["PATH"]
    if python_environment_path:
        environ["PATH"] = str(python_environment_path) + os.pathsep + environ["PATH"]
        logger.debug("Prepending python environment %s to PATH", python_environment_path)

    log_level = logging.getLevelName(logger.getEffectiveLevel())
    environ["OLIVE_LOG_LEVEL"] = log_level
    return environ


def run_available_providers_runner(environ: dict) -> list[str]:
    """Run the available providers runner script with the given environment and return the available providers."""
    runner_path = Path(__file__).parent / "available_providers_runner.py"
    python_path = shutil.which("python", path=environ["PATH"])
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir).resolve() / "available_eps.json"
        run_subprocess(
            f"{python_path} {runner_path} --output_path {output_path}",
            env=environ,
            check=True,
        )
        with output_path.open("r") as f:
            return json.load(f)
