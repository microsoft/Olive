# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from olive.common.utils import hash_dir
from olive.hardware.constants import PROVIDER_DOCKERFILE_MAPPING, PROVIDER_PACKAGE_MAPPING
from olive.systems.common import SystemType

logger = logging.getLogger(__name__)


def get_package_name_from_ep(execution_provider):
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, ("onnxruntime", "ort-nightly"))


@lru_cache(maxsize=8)
def create_new_system_with_cache(system_config, accelerator):
    return create_new_system(system_config, accelerator)


def create_new_system(system_config, accelerator):
    # pylint: disable=consider-using-with

    # create a new system with the same type as the origin system
    if system_config.type == SystemType.Local:
        raise NotImplementedError("olive_managed_env is not supported for LocalSystem")

    elif system_config.type == SystemType.PythonEnvironment:
        import os
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
                "build_context_path": Path(__file__).parent / "docker",
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
        shutil.copy2(str(Path(__file__).parent / "docker" / dockerfile), build_context_path)
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
