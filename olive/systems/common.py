# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from pathlib import Path
from typing import List, Optional, Union

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator


class SystemType(str, Enum):
    Docker = "Docker"
    Local = "LocalSystem"
    AzureML = "AzureML"
    PythonEnvironment = "PythonEnvironment"
    IsolatedORT = "IsolatedORT"


class AcceleratorConfig(ConfigBase):
    device: str = None
    execution_providers: List[str] = None

    @validator("execution_providers", always=True)
    def validate_device_and_execution_providers(cls, v, values):
        if v is None and values.get("device") is None:
            raise ValueError("Either device or execution_providers must be provided")
        return v


class AzureMLDockerConfig(ConfigBase):
    base_image: Optional[str] = None
    dockerfile: Optional[str] = None
    build_context_path: Optional[Union[Path, str]] = None
    conda_file_path: Optional[Union[Path, str]] = None
    name: Optional[str] = None
    version: Optional[str] = None

    @validator("dockerfile", "build_context_path", always=True)
    def _validate_one_of_dockerfile_or_base_image(cls, v, values):
        if v is None and values.get("base_image") is None:
            raise ValueError("One of build_context_path/dockerfile or base_image must be provided")
        return v

    @validator("conda_file_path")
    def _get_abspath(cls, v):
        if v:
            return str(Path(v).resolve())
        else:
            return None


class AzureMLEnvironmentConfig(ConfigBase):
    name: str
    version: Optional[str] = None
    label: Optional[str] = None


class LocalDockerConfig(ConfigBase):
    image_name: str
    dockerfile: Optional[str] = None
    build_context_path: Optional[Union[Path, str]] = None
    build_args: Optional[dict] = None
    run_params: Optional[dict] = None

    @validator("build_context_path")
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None
