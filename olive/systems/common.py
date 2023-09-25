# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from enum import Enum
from pathlib import Path
from typing import Optional, Union

from pydantic import validator

from olive.common.config_utils import ConfigBase


class SystemType(str, Enum):
    Docker = "Docker"
    Local = "LocalSystem"
    AzureML = "AzureML"
    PythonEnvironment = "PythonEnvironment"


class AzureMLDockerConfig(ConfigBase):
    base_image: Optional[str] = None
    dockerfile: Optional[str] = None
    build_context_path: Optional[Union[Path, str]] = None
    conda_file_path: Optional[Union[Path, str]] = None

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


class LocalDockerConfig(ConfigBase):
    image_name: str
    base_image: Optional[str] = None
    requirements_file_path: Optional[str] = None
    dockerfile: Optional[str] = None
    build_context_path: Optional[Union[Path, str]] = None
    build_args: Optional[dict] = None
    run_params: Optional[dict] = None

    @validator("build_context_path", "requirements_file_path")
    def _get_abspath(cls, v):
        return str(Path(v).resolve()) if v else None

    @validator("dockerfile", always=True)
    def _validate_one_of_dockerfile_or_base_image(cls, v, values):
        if v is None and values.get("requirements_file_path") is None:
            raise ValueError("One of build_context_path/dockerfile or requirements_file_path must be provided")
        return v
