# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

from pydantic import FieldValidationInfo, SerializeAsAny, field_validator, model_validator

import olive.systems.system_alias as system_alias
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.systems.common import AzureMLDockerConfig, LocalDockerConfig, SystemType


class TargetUserConfig(ConfigBase):
    accelerators: Optional[List[str]] = None


class LocalTargetUserConfig(TargetUserConfig):
    pass


class DockerTargetUserConfig(TargetUserConfig):
    local_docker_config: LocalDockerConfig
    is_dev: bool = False


class AzureMLTargetUserConfig(TargetUserConfig):
    azureml_client_config: Optional[AzureMLClientConfig] = None
    aml_compute: str
    aml_docker_config: AzureMLDockerConfig
    instance_count: int = 1
    is_dev: bool = False


class PythonEnvironmentTargetUserConfig(TargetUserConfig):
    python_environment_path: Union[
        Path, str
    ]  # path to the python environment, e.g. /home/user/anaconda3/envs/myenv, /home/user/.virtualenvs/myenv
    environment_variables: Optional[Dict[str, str]] = None  # os.environ will be updated with these variables
    prepend_to_path: Optional[List[str]] = None  # paths to prepend to os.environ["PATH"]

    @model_validator(mode="before")
    @classmethod
    def validate_pathes(cls, data):
        python_environment_path = data.get("python_environment_path")
        if python_environment_path:
            data["python_environment_path"] = str(Path(python_environment_path).resolve())

        prepend_to_path = data.get("prepend_to_path")
        if prepend_to_path:
            data["prepend_to_path"] = [str(Path(p).resolve()) for p in prepend_to_path]
        return data

    @field_validator("python_environment_path")
    def _validate_python_environment_path(cls, v):
        # check if the path exists
        if not Path(v).exists():
            raise ValueError(f"Python path {v} does not exist")

        # check if python exists in the path
        python_path = shutil.which("python", path=v)
        if not python_path:
            raise ValueError(f"Python executable not found in the path {v}")
        return v


_type_to_config = {
    SystemType.Local: LocalTargetUserConfig,
    SystemType.AzureML: AzureMLTargetUserConfig,
    SystemType.Docker: DockerTargetUserConfig,
    SystemType.PythonEnvironment: PythonEnvironmentTargetUserConfig,
}

_type_to_system_path = {
    SystemType.Local: "olive.systems.local.LocalSystem",
    SystemType.AzureML: "olive.systems.azureml.AzureMLSystem",
    SystemType.Docker: "olive.systems.docker.DockerSystem",
    SystemType.PythonEnvironment: "olive.systems.python_environment.PythonEnvironmentSystem",
}


def import_system_from_type(system_type: SystemType):
    system_path = _type_to_system_path[system_type]
    module_path, class_name = system_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SystemConfig(ConfigBase):
    type: SystemType
    config: Optional[SerializeAsAny[TargetUserConfig]] = None

    @model_validator(mode="before")
    @classmethod
    def validate_config_type(cls, values):
        type_name = values.get("type")
        system_alias_class = getattr(system_alias, type_name, None)
        if system_alias_class:
            values["type"] = system_alias_class.system_type
            values["config"]["accelerators"] = system_alias_class.accelerators
            # TODO: consider how to use num_cpus and num_gpus in distributed inference.
        return values

    @field_validator("config", mode="before")
    def validate_config(cls, v, info: FieldValidationInfo):
        if "type" not in info.data:
            raise ValueError("Invalid type")

        system_type = info.data["type"]
        config_class = _type_to_config[system_type]
        return validate_config(v, config_class)

    def create_system(self):
        system_class = import_system_from_type(self.type)
        if system_class.system_type == SystemType.AzureML and not self.config.azureml_client_config:
            raise ValueError("azureml_client is required for AzureML system")
        return system_class(**self.config.model_dump())
