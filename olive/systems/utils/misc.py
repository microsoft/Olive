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
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

from olive.common.constants import OS
from olive.common.utils import run_subprocess
from olive.hardware import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
from olive.hardware.constants import PROVIDER_PACKAGE_MAPPING
from olive.systems.common import SystemType

if TYPE_CHECKING:
    from olive.systems.system_config import SystemConfig

logger = logging.getLogger(__name__)


def get_package_name_from_ep(execution_provider: str) -> str:
    """Get the package name from the execution provider."""
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, "onnxruntime")


@lru_cache(maxsize=8)
def create_managed_system_with_cache(system_config, accelerator):
    return create_managed_system(system_config, accelerator)


def create_managed_system(system_config: "SystemConfig", accelerator: "AcceleratorSpec"):
    # pylint: disable=consider-using-with
    assert system_config.olive_managed_env, "system_config.olive_managed_env must be True"

    # for host system, use the first available accelerator
    if accelerator:
        accelerator_cfg = [{"device": accelerator.accelerator_type}]
        if accelerator.execution_provider:
            accelerator_cfg[0]["execution_providers"] = [accelerator.execution_provider]
    else:
        accelerator_cfg = None
        accelerator = DEFAULT_CPU_ACCELERATOR

    # create a new system with the same type as the origin system
    if system_config.type == SystemType.PythonEnvironment:
        import platform
        import venv

        from olive.systems.python_environment import PythonEnvironmentSystem

        if platform.system() == OS.LINUX:
            destination_dir = os.path.join(os.environ.get("HOME", ""), "tmp")
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            venv_path = Path(tempfile.TemporaryDirectory(prefix="olive_python_env_", dir=destination_dir).name)
        else:
            venv_path = Path(tempfile.TemporaryDirectory(prefix="olive_python_env_").name)

        venv.create(venv_path, with_pip=True, system_site_packages=True)
        logger.info("Virtual environment '%s' created.", venv_path)

        if platform.system() == OS.WINDOWS:
            python_environment_path = f"{venv_path}/Scripts"
        else:
            python_environment_path = f"{venv_path}/bin"
        new_system = PythonEnvironmentSystem(
            python_environment_path=python_environment_path,
            accelerators=accelerator_cfg,
            environment_variables=system_config.config.environment_variables,
            prepend_to_path=system_config.config.prepend_to_path,
            olive_managed_env=True,
            requirements_file=system_config.config.requirements_file,
        )
        new_system.install_requirements(accelerator)

    else:
        raise NotImplementedError(f"System type {system_config.type} is not supported")

    return new_system


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
