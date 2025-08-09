# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Optional, Union

from olive.common.config_utils import CaseInsensitiveEnum, ConfigBase
from olive.common.pydantic_v1 import Field, validator
from olive.hardware.accelerator import AcceleratorSpec, Device


class SystemType(CaseInsensitiveEnum):
    Docker = "Docker"
    Local = "LocalSystem"
    AzureML = "AzureML"
    PythonEnvironment = "PythonEnvironment"
    IsolatedORT = "IsolatedORT"


class AcceleratorConfig(ConfigBase):
    device: Union[str, Device] = Field(None, description="Device to use for the accelerator")
    execution_providers: list[Union[str, tuple[str, str]]] = Field(
        None,
        description=(
            "Execution providers for the accelerator. Each must end with ExecutionProvider. If a tuple, the second"
            " element is the path to the EP library. Tuple is only supported for Local System."
        ),
    )
    memory: Union[int, str] = Field(
        None, description="Memory size of accelerator in bytes. Can also be provided in string format like 1GB."
    )

    @validator("execution_providers", always=True)
    def validate_device_and_execution_providers(cls, v, values):
        if not v and values.get("device") is None:
            # checking for not v since v could be an empty list
            raise ValueError("Either device or execution_providers must be provided")
        return v

    @validator("execution_providers", pre=True, each_item=True)
    def validate_ep_suffix(cls, v):
        if not v:
            return v

        ep_name = v[0] if isinstance(v, (tuple, list)) else v
        if not ep_name.endswith("ExecutionProvider"):
            raise ValueError(f"Execution provider {ep_name} should end with ExecutionProvider")
        return v

    @validator("memory", pre=True)
    def validate_memory(cls, v):
        return AcceleratorSpec.str_to_int_memory(v)

    def get_ep_strs(self) -> Optional[list[str]]:
        """Get execution provider names as strings."""
        return (
            [ep[0] if isinstance(ep, tuple) else ep for ep in self.execution_providers]
            if self.execution_providers
            else None
        )

    def get_ep_path_map(self) -> dict[str, str]:
        """Get a map of execution provider names to their library paths."""
        ep_path_map = {}
        for ep in self.execution_providers or []:
            if isinstance(ep, tuple):
                ep_path_map[ep[0]] = ep[1]
            elif ep != "CPUExecutionProvider":
                # CPU EP is built-in and does not have a path
                ep_path_map[ep] = None
        return ep_path_map


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
