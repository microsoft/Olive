# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from pathlib import Path
from typing import Union

from pydantic import validator

from olive.common.config_utils import ConfigBase, validate_config
from olive.systems.common import AzureMLDockerConfig, Device, LocalDockerConfig, SystemType


class TargetUserConfig(ConfigBase):
    device: Device = Device.CPU


class LocalTargetUserConfig(TargetUserConfig):
    pass


class DockerTargetUserConfig(TargetUserConfig):
    local_docker_config: LocalDockerConfig


class AzureMLTargetUserConfig(TargetUserConfig):
    aml_config_path: Union[Path, str]
    aml_compute: str
    aml_docker_config: AzureMLDockerConfig
    instance_count: int = 1
    is_dev: bool = False


_type_to_config = {
    SystemType.Local: LocalTargetUserConfig,
    SystemType.AzureML: AzureMLTargetUserConfig,
    SystemType.Docker: DockerTargetUserConfig,
}

_type_to_system_path = {
    SystemType.Local: "olive.systems.local.LocalSystem",
    SystemType.AzureML: "olive.systems.azureml.AzureMLSystem",
    SystemType.Docker: "olive.systems.docker.DockerSystem",
}


def import_system_from_type(system_type: SystemType):
    system_path = _type_to_system_path[system_type]
    module_path, class_name = system_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SystemConfig(ConfigBase):
    type: SystemType
    config: TargetUserConfig = None

    @validator("config", pre=True, always=True)
    def validate_config(cls, v, values):
        if "type" not in values:
            raise ValueError("Invalid type")

        system_type = values["type"]
        config_class = _type_to_config[system_type]
        return validate_config(v, config_class)

    def create_system(self):
        system_class = import_system_from_type(self.type)
        return system_class(**self.config.dict())
