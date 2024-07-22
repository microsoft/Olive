# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import shutil
from pathlib import Path
from typing import Dict, List, Union

import olive.systems.system_alias as system_alias
from olive.azureml.azureml_client import AzureMLClientConfig
from olive.common.config_utils import ConfigBase, NestedConfig, validate_config
from olive.common.pydantic_v1 import root_validator, validator
from olive.systems.common import (
    AcceleratorConfig,
    AzureMLDockerConfig,
    AzureMLEnvironmentConfig,
    LocalDockerConfig,
    SystemType,
)


class TargetUserConfig(ConfigBase):
    accelerators: List[AcceleratorConfig] = None
    hf_token: bool = None

    class Config:
        validate_assignment = True


class LocalTargetUserConfig(TargetUserConfig):
    pass


class DockerTargetUserConfig(TargetUserConfig):
    # the local_docker_config is optional for managed environments and required for normal docker system
    local_docker_config: LocalDockerConfig = None
    is_dev: bool = False
    olive_managed_env: bool = False
    requirements_file: Union[Path, str] = None


class AzureMLTargetUserConfig(TargetUserConfig):
    azureml_client_config: AzureMLClientConfig = None
    aml_compute: str
    aml_docker_config: AzureMLDockerConfig = None
    aml_environment_config: AzureMLEnvironmentConfig = None
    tags: Dict = None
    datastores: str = "workspaceblobstore"
    resources: Dict = None
    instance_count: int = 1
    is_dev: bool = False
    olive_managed_env: bool = False
    requirements_file: Union[Path, str] = None


class CommonPythonEnvTargetUserConfig(TargetUserConfig):
    # path to the python environment, e.g. /home/user/anaconda3/envs/myenv, /home/user/.virtualenvs/
    python_environment_path: Union[Path, str] = None
    environment_variables: Dict[str, str] = None  # os.environ will be updated with these variables
    prepend_to_path: List[str] = None  # paths to prepend to os.environ["PATH"]

    @validator("python_environment_path", "prepend_to_path", pre=True, each_item=True)
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None


class PythonEnvironmentTargetUserConfig(CommonPythonEnvTargetUserConfig):
    olive_managed_env: bool = False  # if True, the environment will be created and managed by Olive
    requirements_file: Union[Path, str] = None  # path to the requirements.txt file

    @root_validator(pre=True)
    def _validate_python_environment_path(cls, values):
        # if olive_managed_env is True, python_environment_path is not required
        if values.get("olive_managed_env"):
            return values

        python_environment_path = values.get("python_environment_path")
        if python_environment_path is None:
            raise ValueError("python_environment_path is required for PythonEnvironmentSystem native mode")

        # check if the path exists
        if not Path(python_environment_path).exists():
            raise ValueError(f"Python path {python_environment_path} does not exist")

        # check if python exists in the path
        python_path = shutil.which("python", path=python_environment_path)
        if not python_path:
            raise ValueError(f"Python executable not found in the path {python_environment_path}")
        return values


class IsolatedORTTargetUserConfig(CommonPythonEnvTargetUserConfig):
    # Please refer to https://github.com/pydantic/pydantic/issues/1223
    # In Pydantic v1, missing a optional field will skip the validation. But if the field is specified as None
    # The validation will be triggered. As the result, we cannot use the following line to make the field as required
    # since the validation will still be triggered if user pass it as None.
    # A better approach is to use always=True to check it is required.
    # python_environment_path: Union[Path, str]
    @validator("python_environment_path", always=True)
    def _validate_python_environment_path(cls, v):
        if v is None:
            raise ValueError("python_environment_path is required for IsolatedORTSystem")

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
    SystemType.IsolatedORT: IsolatedORTTargetUserConfig,
}

_type_to_system_path = {
    SystemType.Local: "olive.systems.local.LocalSystem",
    SystemType.AzureML: "olive.systems.azureml.AzureMLSystem",
    SystemType.Docker: "olive.systems.docker.DockerSystem",
    SystemType.PythonEnvironment: "olive.systems.python_environment.PythonEnvironmentSystem",
    SystemType.IsolatedORT: "olive.systems.isolated_ort.IsolatedORTSystem",
}


def import_system_from_type(system_type: SystemType):
    system_path = _type_to_system_path[system_type]
    module_path, class_name = system_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SystemConfig(NestedConfig):
    type: SystemType
    config: TargetUserConfig = None

    @root_validator(pre=True)
    def validate_config_type(cls, values):
        type_name = values.get("type")
        system_alias_class = getattr(system_alias, type_name, None)
        if system_alias_class:
            values["type"] = system_alias_class.system_type
            if "config" not in values:
                values["config"] = {}

            if values["type"] == SystemType.AzureML and not values["config"].get("accelerators"):
                raise ValueError("accelerators is required for AzureML system")

            if system_alias_class.accelerators:
                valid_accelerators = []

                if not values["config"].get("accelerators"):
                    valid_accelerators = [
                        {"device": acc, "execution_providers": None} for acc in system_alias_class.accelerators
                    ]
                else:
                    for device in system_alias_class.accelerators:
                        valid_accelerators.extend(
                            {"device": acc["device"], "execution_providers": acc.get("execution_providers")}
                            for acc in values["config"]["accelerators"]
                            if acc["device"].lower() == device.lower()
                        )

                values["config"]["accelerators"] = valid_accelerators or None
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

    @property
    def olive_managed_env(self):
        return getattr(self.config, "olive_managed_env", False)

    # the __hash__ is needed so to create_managed_system_with_cache, otherwise the following error will be raised:
    # unhashable type: 'SystemConfig'
    __hash__ = object.__hash__
