# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import yaml

from olive.systems.common import SystemType


def parse_common_args(raw_args):
    parser = argparse.ArgumentParser("Olive model args")

    # model args
    parser.add_argument("--model_config", type=str, help="model config", required=True)
    parser.add_argument("--model_path", type=str, help="model path")
    parser.add_argument("--model_script", type=str, help="model script")
    parser.add_argument("--model_script_dir", type=str, help="model script dir")

    # pipeline output arg
    # model output args
    parser.add_argument("--pipeline_output", type=str, help="pipeline output path", required=True)

    return parser.parse_known_args(raw_args)


def get_model_config(common_args):
    with open(common_args.model_config) as f:
        model_json = json.load(f)

    for key, value in common_args.__dict__.items():
        if value and key in model_json["config"]:
            model_json["config"][key] = value

    return model_json


def get_package_name(execution_provider):
    PROVIDER_PACKAGE_MAPPING = {
        "CPUExecutionProvider": "onnxruntime",
        "CUDAExecutionProvider": "onnxruntime-gpu",
        "TensorrtExecutionProvider": "onnxruntime-gpu",
        "OpenVINOExecutionProvider": "onnxruntime-openvino",
    }
    return PROVIDER_PACKAGE_MAPPING.get(execution_provider, "onnxruntime")


def create_new_system(origin_system, accelerator):
    PROVIDER_DOCKERFILE_MAPPING = {
        "CPUExecutionProvider": "Dockerfile.cpu",
        "CUDAExecutionProvider": "Dockerfile.gpu",
        "TensorrtExecutionProvider": "Dockerfile.gpu",
        "OpenVINOExecutionProvider": "Dockerfile.openvino",
    }

    # create a new system with the same type as the origin system
    if origin_system.system_type == SystemType.Local:
        raise NotImplementedError("olive_managed_env is not supported for LocalSystem")

    elif origin_system.system_type == SystemType.PythonEnvironment:
        import tempfile
        import venv

        from olive.systems.python_environment import PythonEnvironmentSystem

        # Create the virtual environment
        venv_path = Path(tempfile.TemporaryDirectory(prefix="olive_python_env_").name)
        venv.create(venv_path, with_pip=True)
        import platform

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

        dockerfile = PROVIDER_DOCKERFILE_MAPPING.get(accelerator.execution_provider, "Dockerfile.cpu")
        new_system = DockerSystem(
            local_docker_config={
                "image_name": f"olive_{accelerator.execution_provider[:-17].lower()}",
                "requirements_file_path": str(origin_system.requirements_file),
                "dockerfile": dockerfile,
                "build_context_path": Path(__file__).parent / "docker",
            },
            accelerators=[accelerator.accelerator_type],
        )

    elif origin_system.system_type == SystemType.AzureML:
        from olive.systems.azureml import AzureMLSystem

        dockerfile = PROVIDER_DOCKERFILE_MAPPING.get(accelerator.execution_provider, "Dockerfile.cpu")
        conda_file_path = export_conda_yaml_from_requirements(origin_system.requirements_file)
        new_system = AzureMLSystem(
            azureml_client_config=origin_system.azureml_client_config,
            aml_compute=origin_system.compute,
            instance_count=origin_system.instance_count,
            accelerators=[accelerator.accelerator_type],
            aml_docker_config={
                "dockerfile": dockerfile,
                "conda_file_path": conda_file_path,
                "build_context_path": Path(__file__).parent / "docker",
            },
        )

    else:
        raise NotImplementedError(f"System type {origin_system.system_type} is not supported")

    return new_system


def export_conda_yaml_from_requirements(requirements_file):
    temp_dir = tempfile.TemporaryDirectory(prefix="olive_")
    conda_file = os.path.join(temp_dir.name, "environment.yml")
    subprocess.run(["conda", "env", "export", "-f", conda_file, "--no-builds"], check=True)
    with open(conda_file) as f:
        conda_env = yaml.safe_load(f)
    conda_env["dependencies"] = conda_env["dependencies"] + [{"pip": requirements_file}]
    with open(conda_file, "w") as f:
        yaml.dump(conda_env, f)
    return conda_file
