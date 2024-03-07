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
from typing import Dict, List, Optional, Union

from olive.common.utils import hash_dir, run_subprocess
from olive.hardware.constants import PROVIDER_DOCKERFILE_MAPPING, PROVIDER_PACKAGE_MAPPING
from olive.systems.common import SystemType

logger = logging.getLogger(__name__)


def get_package_name_from_ep(execution_provider: str) -> str:
    """Get the package name from the execution provider."""
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, ("onnxruntime", "ort-nightly"))


@lru_cache(maxsize=8)
def create_new_system_with_cache(system_config, accelerator):
    return create_new_system(system_config, accelerator)


def create_new_system(system_config, accelerator):
    # pylint: disable=consider-using-with

    # create a new system with the same type as the origin system
    if system_config.type in [SystemType.Local, SystemType.IsolatedORT]:
        raise NotImplementedError(f"olive_managed_env is not supported for {system_config.type} System")

    elif system_config.type == SystemType.PythonEnvironment:
        import platform
        import venv

        from olive.systems.python_environment import PythonEnvironmentSystem

        if platform.system() == "Linux":
            destination_dir = os.path.join(os.environ.get("HOME", ""), "tmp")
            if not os.path.exists(destination_dir):
                os.makedirs(destination_dir)
            venv_path = Path(tempfile.TemporaryDirectory(prefix="olive_python_env_", dir=destination_dir).name)
        else:
            venv_path = Path(tempfile.TemporaryDirectory(prefix="olive_python_env_").name)

        venv.create(venv_path, with_pip=True, system_site_packages=True)
        logger.info("Virtual environment '%s' created.", venv_path)

        if platform.system() == "Windows":
            python_environment_path = f"{venv_path}/Scripts"
        else:
            python_environment_path = f"{venv_path}/bin"
        new_system = PythonEnvironmentSystem(
            python_environment_path=python_environment_path,
            accelerators=[accelerator.accelerator_type],
            environment_variables=system_config.config.environment_variables,
            prepend_to_path=system_config.config.prepend_to_path,
            olive_managed_env=True,
            requirements_file=system_config.config.requirements_file,
        )
        new_system.install_requirements(accelerator)

    elif system_config.type == SystemType.Docker:
        from olive.systems.docker import DockerSystem

        dockerfile = PROVIDER_DOCKERFILE_MAPPING.get(accelerator.execution_provider, "Dockerfile.cpu")
        # TODO(myguo): create a temp dir for the build context
        new_system = DockerSystem(
            local_docker_config={
                "image_name": f"olive_{accelerator.execution_provider[:-17].lower()}",
                "dockerfile": dockerfile,
                "build_context_path": Path(__file__).parents[1] / "docker",
            },
            accelerators=[accelerator.accelerator_type],
            is_dev=system_config.config.is_dev,
            olive_managed_env=True,
            requirements_file=(
                str(system_config.config.requirements_file) if system_config.config.requirements_file else None
            ),
        )

    elif system_config.type == SystemType.AzureML:
        from olive.systems.azureml import AzureMLSystem

        dockerfile = PROVIDER_DOCKERFILE_MAPPING.get(accelerator.execution_provider, "Dockerfile.cpu")
        temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
        build_context_path = Path(temp_dir.name)
        shutil.copy2(str(Path(__file__).parents[1] / "docker" / dockerfile), build_context_path)
        if system_config.config.requirements_file:
            shutil.copyfile(system_config.config.requirements_file, build_context_path / "requirements.txt")
        else:
            with (build_context_path / "requirements.txt").open("w"):
                pass

        env_hash = hash_dir(build_context_path)
        name = f"olive-managed-env-{env_hash}"
        new_system = AzureMLSystem(
            azureml_client_config=system_config.config.azureml_client_config,
            aml_compute=system_config.config.aml_compute,
            instance_count=system_config.config.instance_count,
            accelerators=[accelerator.accelerator_type],
            aml_environment_config={
                "name": name,
                "label": "latest",
            },
            aml_docker_config={
                "name": name,
                "dockerfile": dockerfile,
                "build_context_path": build_context_path,
            },
            is_dev=system_config.config.is_dev,
            olive_managed_env=True,
        )
        new_system.temp_dirs.append(temp_dir)

    else:
        raise NotImplementedError(f"System type {system_config.type} is not supported")

    return new_system


def create_new_environ(
    environment_variables: Optional[Dict[str, str]] = None,
    prepend_to_path: Optional[List[str]] = None,
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


def run_available_providers_runner(environ: Dict) -> List[str]:
    """Run the available providers runner script with the given environment and return the available providers."""
    runner_path = Path(__file__).parent / "available_providers_runner.py"
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = Path(temp_dir).resolve() / "available_eps.json"
        run_subprocess(
            f"python {runner_path} --output_path {output_path}",
            env=environ,
            check=True,
        )
        with output_path.open("r") as f:
            return json.load(f)
