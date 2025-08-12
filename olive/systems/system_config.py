# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from pathlib import Path
from typing import Optional, Union

from olive.common.config_utils import ConfigBase, NestedConfig, validate_config
from olive.common.pydantic_v1 import validator
from olive.systems.common import AcceleratorConfig, SystemType


class TargetUserConfig(ConfigBase):
    accelerators: list[AcceleratorConfig] = None
    hf_token: bool = None

    class Config:
        validate_assignment = True


class LocalTargetUserConfig(TargetUserConfig):
    pass


class DockerTargetUserConfig(TargetUserConfig):
    dockerfile: str
    build_context_path: Union[Path, str]
    image_name: str = "olive-docker:latest"
    work_dir: str = "/olive-ws"
    build_args: Optional[dict] = None
    run_params: Optional[dict] = None
    clean_image: bool = True

    @validator("build_context_path")
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None

    @validator("work_dir")
    def _validate_work_dir(cls, v):
        if not v.startswith("/"):
            raise ValueError(f"work_dir must be an absolute path, got: {v}")
        return v


class PythonEnvironmentTargetUserConfig(TargetUserConfig):
    # path to the python environment, e.g. /home/user/anaconda3/envs/myenv, /home/user/.virtualenvs/
    python_environment_path: Union[Path, str] = None
    environment_variables: dict[str, str] = None  # os.environ will be updated with these variables
    prepend_to_path: list[str] = None  # paths to prepend to os.environ["PATH"]

    @validator("python_environment_path", "prepend_to_path", pre=True, each_item=True)
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None


_type_to_config = {
    SystemType.Local: LocalTargetUserConfig,
    SystemType.Docker: DockerTargetUserConfig,
    SystemType.PythonEnvironment: PythonEnvironmentTargetUserConfig,
}

_type_to_system_path = {
    SystemType.Local: "olive.systems.local.LocalSystem",
    SystemType.Docker: "olive.systems.docker.DockerSystem",
    SystemType.PythonEnvironment: "olive.systems.python_environment.PythonEnvironmentSystem",
}


def import_system_from_type(system_type: SystemType):
    system_path = _type_to_system_path[system_type]
    module_path, class_name = system_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


class SystemConfig(NestedConfig):
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
