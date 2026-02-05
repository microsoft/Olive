# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Optional, Union

from pydantic import Field, field_validator, model_validator

from olive.common.config_utils import CaseInsensitiveEnum, ConfigBase
from olive.hardware.accelerator import AcceleratorSpec, Device


class SystemType(CaseInsensitiveEnum):
    Docker = "Docker"
    Local = "LocalSystem"
    PythonEnvironment = "PythonEnvironment"


class AcceleratorConfig(ConfigBase):
    device: Optional[Union[str, Device]] = Field(None, description="Device to use for the accelerator")
    execution_providers: Optional[list[Union[str, tuple[str, str]]]] = Field(
        None,
        description=(
            "Execution providers for the accelerator. Each must end with ExecutionProvider. If a tuple, the second"
            " element is the path to the EP library. Tuple is only supported for Local System."
        ),
        validate_default=True,
    )
    memory: Optional[Union[int, str]] = Field(
        None, description="Memory size of accelerator in bytes. Can also be provided in string format like 1GB."
    )

    @model_validator(mode="after")
    def validate_device_and_execution_providers(self):  # noqa: N804  # model_validator mode="after" uses self
        if not self.execution_providers and self.device is None:
            # checking for not execution_providers since it could be an empty list
            raise ValueError("Either device or execution_providers must be provided")
        if self.execution_providers and len(self.execution_providers) > 1:
            raise ValueError("Only one execution provider is supported per accelerator")
        return self

    @field_validator("execution_providers", mode="before")
    @classmethod
    def validate_ep_suffix(cls, v):
        if not v:
            return v

        result = []
        for item in (v if isinstance(v, list) else [v]):
            ep_name = item[0] if isinstance(item, (tuple, list)) else item
            if not ep_name.endswith("ExecutionProvider"):
                raise ValueError(f"Execution provider {ep_name} should end with ExecutionProvider")
            result.append(item)
        return result

    @field_validator("memory", mode="before")
    @classmethod
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
