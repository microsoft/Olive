# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from pathlib import Path
from typing import Optional, Union

from pydantic import ConfigDict, Field, field_validator

from olive.common.config_utils import ConfigBase, NestedConfig, validate_config
from olive.systems.common import AcceleratorConfig, SystemType


class TargetUserConfig(ConfigBase):
    model_config = ConfigDict(validate_assignment=True)

    accelerators: Optional[list[AcceleratorConfig]] = None
    hf_token: Optional[bool] = None

    @field_validator("accelerators", mode="before")
    @classmethod
    def validate_accelerators(cls, v):
        if v and len(v) > 1:
            raise ValueError("Only one accelerator is supported currently.")
        return v


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

    @field_validator("build_context_path")
    @classmethod
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None

    @field_validator("work_dir")
    @classmethod
    def _validate_work_dir(cls, v):
        if not v.startswith("/"):
            raise ValueError(f"work_dir must be an absolute path, got: {v}")
        return v


class PythonEnvironmentTargetUserConfig(TargetUserConfig):
    # path to the python environment, e.g. /home/user/anaconda3/envs/myenv, /home/user/.virtualenvs/
    python_environment_path: Optional[Union[Path, str]] = None
    environment_variables: Optional[dict[str, str]] = None  # os.environ will be updated with these variables
    prepend_to_path: Optional[list[str]] = None  # paths to prepend to os.environ["PATH"]

    @field_validator("python_environment_path", "prepend_to_path", mode="before")
    @classmethod
    def _get_abspath(cls, v):
        if isinstance(v, list):
            return [str(Path(item).resolve()) if item else None for item in v]
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
    config: Optional[TargetUserConfig] = Field(default = None, validate_default=True)

    @field_validator("config", mode="before")
    @classmethod
    def validate_config(cls, v, info):
        if "type" not in info.data:
            raise ValueError("Invalid type")

        config_class = _type_to_config[info.data["type"]]
        return validate_config(v, config_class)

    def create_system(self):
        system_class = import_system_from_type(self.type)
        config_dict = self.config.model_dump() if self.config else {}
        return system_class(**config_dict)
