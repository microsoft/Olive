# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import logging
import shutil
import tempfile
from functools import lru_cache
from pathlib import Path

from olive.systems.common import SystemType

logger = logging.getLogger(__name__)


def parse_common_args(raw_args):
    """Parse common args."""
    parser = argparse.ArgumentParser("Olive common args")

    # model args
    parser.add_argument("--model_config", type=str, help="Path to model config json", required=True)

    # pipeline output arg
    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    return parser.parse_known_args(raw_args)


def parse_model_resources(model_resource_names, raw_args):
    """Parse model resources. These are the resources that were uploaded with the job."""
    parser = argparse.ArgumentParser("Model resources")

    # parse model resources
    for resource_name in model_resource_names:
        # will keep as required since only uploaded resources should be in this list
        parser.add_argument(f"--model_{resource_name}", type=str, required=True)

    return parser.parse_known_args(raw_args)


def get_common_args(raw_args):
    """Return the model_config.

    The return value includes json with the model resource paths filled in, the pipeline output path, and any
    extra args that were not parsed.
    """
    common_args, extra_args = parse_common_args(raw_args)

    # load model json
    with open(common_args.model_config) as f:  # noqa: PTH123
        model_json = json.load(f)
    # model json has a list of model resource names
    model_resource_names = model_json.pop("resource_names")
    model_resource_args, extra_args = parse_model_resources(model_resource_names, extra_args)

    for key, value in vars(model_resource_args).items():
        # remove the model_ prefix, the 1 is to only replace the first occurrence
        normalized_key = key.replace("model_", "", 1)
        model_json["config"][normalized_key] = value

    return model_json, common_args.pipeline_output, extra_args


@lru_cache(maxsize=8)
def create_new_system_with_cache(origin_system, accelerator):
    return create_new_system(origin_system, accelerator)


def create_new_system(origin_system, accelerator):
    provider_dockerfile_mapping = {
        "CPUExecutionProvider": "Dockerfile.cpu",
        "CUDAExecutionProvider": "Dockerfile.gpu",
        "TensorrtExecutionProvider": "Dockerfile.gpu",
        "OpenVINOExecutionProvider": "Dockerfile.openvino",
    }

    # create a new system with the same type as the origin system
    if origin_system.system_type == SystemType.Local:
        raise NotImplementedError("olive_managed_env is not supported for LocalSystem")

    elif origin_system.system_type == SystemType.PythonEnvironment:
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
        logger.info("Virtual environment '{}' created.".format(venv_path))

        if platform.system() == "Windows":
            python_environment_path = f"{venv_path}/Scripts"
        else:
            python_environment_path = f"{venv_path}/bin"
        new_system = PythonEnvironmentSystem(
            python_environment_path=python_environment_path,
            accelerators=[accelerator.accelerator_type],
            environment_variables=origin_system.config.environment_variables,
            prepend_to_path=origin_system.config.prepend_to_path,
            olive_managed_env=True,
            requirements_file=origin_system.config.requirements_file,
        )
        new_system.install_requirements(accelerator)

    elif origin_system.system_type == SystemType.Docker:
        from olive.systems.docker import DockerSystem

        dockerfile = provider_dockerfile_mapping.get(accelerator.execution_provider, "Dockerfile.cpu")
        new_system = DockerSystem(
            local_docker_config={
                "image_name": f"olive_{accelerator.execution_provider[:-17].lower()}",
                "requirements_file_path": str(origin_system.requirements_file),
                "dockerfile": dockerfile,
                "build_context_path": Path(__file__).parent / "docker",
            },
            accelerators=[accelerator.accelerator_type],
            is_dev=origin_system.is_dev,
        )

    elif origin_system.system_type == SystemType.AzureML:
        from olive.systems.azureml import AzureMLSystem

        dockerfile = provider_dockerfile_mapping.get(accelerator.execution_provider, "Dockerfile.cpu")
        build_context_path = Path(__file__).parent / "docker"

        if origin_system.requirements_file:
            shutil.copyfile(origin_system.requirements_file, build_context_path / "requirements.txt")
        else:
            with (build_context_path / "requirements.txt").open("w"):
                pass

        new_system = AzureMLSystem(
            azureml_client_config=origin_system.azureml_client_config,
            aml_compute=origin_system.compute,
            instance_count=origin_system.instance_count,
            accelerators=[accelerator.accelerator_type],
            aml_docker_config={
                "dockerfile": dockerfile,
                "build_context_path": build_context_path,
            },
            is_dev=origin_system.is_dev,
        )

    else:
        raise NotImplementedError(f"System type {origin_system.system_type} is not supported")

    return new_system
