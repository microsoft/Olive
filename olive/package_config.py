# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from pathlib import Path
from typing import Dict, List

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import validator
from olive.passes import PassModuleConfig


class OlivePackageConfig(ConfigBase):
    """Configuration for an Olive package.

    passes key is case-insensitive and stored in lowercase.
    """

    passes: Dict[str, PassModuleConfig]
    extra_dependencies: Dict[str, List[str]]

    @validator("passes")
    def validate_passes(cls, values):
        return {key.lower(): value for key, value in values.items()}

    @staticmethod
    def get_default_config_path() -> str:
        return str(Path(__file__).parent / "olive_config.json")

    @staticmethod
    def load_default_config() -> "OlivePackageConfig":
        return OlivePackageConfig.parse_file(OlivePackageConfig.get_default_config_path())

    def import_pass_module(self, pass_type: str):
        pass_module_config = self.get_pass_module_config(pass_type)
        module_path, module_name = pass_module_config.module_path.rsplit(".", 1)
        module = importlib.import_module(module_path, module_name)
        return getattr(module, module_name)

    def get_pass_module_config(self, pass_type: str) -> PassModuleConfig:
        if "." in pass_type:
            _, module_name = pass_type.rsplit
            return self.get_pass_module_config(module_name)

        pass_type = pass_type.lower()
        if pass_type in self.passes:
            return self.passes.get(pass_type)

        raise ValueError(f"Package configuration for pass of type '{pass_type}' not found")
