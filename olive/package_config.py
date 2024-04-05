# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
from pathlib import Path
from typing import Dict, List

from olive.common.config_utils import ConfigBase
from olive.passes import PassModuleConfig


class OlivePackageConfig(ConfigBase):
    passes: Dict[str, PassModuleConfig]
    extra_dependencies: Dict[str, List[str]]

    @staticmethod
    def get_default_config_path() -> str:
        return str(Path(__file__).parent / "olive_config.json")

    @staticmethod
    def load_default_config() -> "OlivePackageConfig":
        return OlivePackageConfig.parse_file(OlivePackageConfig.get_default_config_path())

    def import_pass_module(self, pass_type):
        if "." in pass_type:
            _, module_name = pass_type.rsplit(".", 1)
            return self.import_pass_module(module_name)

        if pass_type in self.passes:
            pass_module_config = self.passes.get(pass_type)
            module_path, module_name = pass_module_config.module_path.rsplit(".", 1)
            module = importlib.import_module(module_path, module_name)
            return getattr(module, module_name)

        raise ValueError(f"Package configuration for pass of type '{pass_type}' not found")
