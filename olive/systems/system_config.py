# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import shutil
from pathlib import Path
from typing import Dict, List, Union

from pydantic import root_validator, validator

import olive.systems.system_alias as system_alias
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, validate_config
from olive.systems.common import AzureMLDockerConfig, LocalDockerConfig, SystemType


class TargetUserConfig(ConfigBase):
    accelerators: List[str] = None


class LocalTargetUserConfig(TargetUserConfig):
    pass


class DockerTargetUserConfig(TargetUserConfig):
    local_docker_config: LocalDockerConfig = None
    is_dev: bool = False
    olive_managed_env: bool = False
    requirements_file: Union[Path, str] = None


class AzureMLTargetUserConfig(TargetUserConfig):
    azureml_client_config: AzureMLClientConfig = None
    aml_compute: str
    aml_docker_config: AzureMLDockerConfig = None
    resources: Dict = None
    instance_count: int = 1
    is_dev: bool = False
    olive_managed_env: bool = False
    requirements_file: Union[Path, str] = None


class PythonEnvironmentTargetUserConfig(TargetUserConfig):
    python_environment_path: Union[
        Path, str
    ] = None  # path to the python environment, e.g. /home/user/anaconda3/envs/myenv, /home/user/.virtualenvs/myenv
    environment_variables: Dict[str, str] = None  # os.environ will be updated with these variables
    prepend_to_path: List[str] = None  # paths to prepend to os.environ["PATH"]
    olive_managed_env: bool = False  # if True, the environment will be created and managed by Olive
    requirements_file: Union[Path, str] = None  # path to the requirements.txt file

    @validator("python_environment_path", "prepend_to_path", pre=True, each_item=True)
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None

    @validator("python_environment_path")
    def _validate_python_environment_path(cls, v):
        if v:
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
    type: SystemType  # noqa: A003
    config: TargetUserConfig = None

    @root_validator(pre=True)
    def validate_config_type(cls, values):
        type_name = values.get("type")
        system_alias_class = getattr(system_alias, type_name, None)
        if system_alias_class:
            values["type"] = system_alias_class.system_type
            values["config"]["accelerators"] = system_alias_class.accelerators
            # TODO(myguo): consider how to use num_cpus and num_gpus in distributed inference.
        return values

    @validator("config", pre=True, always=True)
    def validate_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        system_type = values["type"]
        config_class = _type_to_config[system_type]
        return validate_config(v, config_class)

    def create_system(self):
        system_class = import_system_from_type(self.type)
        if system_class.system_type == SystemType.AzureML and not self.config.azureml_client_config:
            raise ValueError("azureml_client is required for AzureML system")
        return system_class(**self.config.dict())
