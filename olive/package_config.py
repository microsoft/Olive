# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import importlib
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar, Dict, List, Type

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field, validator
from olive.passes import PassModuleConfig

if TYPE_CHECKING:
    from olive.passes.olive_pass import Pass


class OlivePackageConfig(ConfigBase):
    """Configuration for an Olive package.

    passes key is case-insensitive and stored in lowercase.
    """

    passes: Dict[str, PassModuleConfig] = Field(default_factory=dict)
    extra_dependencies: Dict[str, List[str]] = Field(default_factory=dict)

    _pass_modules: ClassVar[Dict[str, Type["Pass"]]] = {}

    @validator("passes")
    def validate_passes(cls, values):
        return {key.lower(): value for key, value in values.items()}

    @staticmethod
    @functools.lru_cache
    def get_default_config_path() -> str:
        return str(Path(__file__).parent / "olive_config.json")

    @staticmethod
    @functools.lru_cache
    def load_default_config() -> "OlivePackageConfig":
        return OlivePackageConfig.parse_file(OlivePackageConfig.get_default_config_path())

    def import_pass_module(self, pass_type: str):
        if pass_type not in self._pass_modules:
            pass_module_config = self.get_pass_module_config(pass_type)
            module_path, module_name = pass_module_config.module_path.rsplit(".", 1)
            module = importlib.import_module(module_path, module_name)
            self._pass_modules[pass_type] = cls = getattr(module, module_name)
            pass_module_config.set_class_variables(cls)
        return self._pass_modules[pass_type]

    def get_pass_module_config(self, pass_type: str) -> PassModuleConfig:
        if "." in pass_type:
            _, module_name = pass_type.rsplit
            return self.get_pass_module_config(module_name)

        pass_type = pass_type.lower()
        if pass_type in self.passes:
            return self.passes.get(pass_type)

        raise ValueError(f"Package configuration for pass of type '{pass_type}' not found")
