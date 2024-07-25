# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

from olive.package_config import OlivePackageConfig


class TestPackageConfig:
    def test_passes_configuration(self):
        package_config = OlivePackageConfig.load_default_config()
        for pass_module_name, pass_module_config in package_config.passes.items():
            assert pass_module_config.module_path
            assert pass_module_config.module_path[-len(pass_module_name) :].lower() == pass_module_name
            package_config.import_pass_module(pass_module_name)
