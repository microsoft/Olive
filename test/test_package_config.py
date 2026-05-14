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

    def test_get_pass_module_config_supports_dotted_path(self):
        package_config = OlivePackageConfig.load_default_config()

        pass_module_config = package_config.get_pass_module_config("olive.passes.onnx.inc_quantization.IncQuantization")

        assert pass_module_config.module_path == "olive.passes.onnx.inc_quantization.IncQuantization"

    def test_inc_quantization_supported_algorithms(self):
        package_config = OlivePackageConfig.load_default_config()

        pass_module_config = package_config.get_pass_module_config("IncQuantization")

        assert pass_module_config.supported_algorithms == {"gptq"}
