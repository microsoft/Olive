# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform

import pytest

from olive.common.constants import OS
from olive.systems.system_config import PythonEnvironmentTargetUserConfig, SystemConfig
from test.unit_test.utils import create_onnx_model_file, get_custom_metric, get_onnx_model_config

# pylint: disable=attribute-defined-outside-init


class TestOliveManagedPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        create_onnx_model_file()
        self.input_model_config = get_onnx_model_config()

    @pytest.mark.skip(reason="No machine to test DML execution provider")
    def test_run_pass_evaluate_windows(self, tmp_path):
        # use the olive managed python environment as the test environment

        from test.multiple_ep.utils import create_and_run_workflow

        system_config = SystemConfig(
            type="PythonEnvironment",
            config=PythonEnvironmentTargetUserConfig(
                accelerators=[
                    {"device": "gpu", "execution_providers": ["DmlExecutionProvider", "OpenVINOExecutionProvider"]}
                ],
                olive_managed_env=True,
            ),
        )
        dml_res, openvino_res = create_and_run_workflow(
            tmp_path, system_config, self.input_model_config, get_custom_metric()
        )
        assert dml_res.metrics.value.__root__
        assert openvino_res.metrics.value.__root__

    @pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="Test for Linux only")
    def test_run_pass_evaluate_linux(self, tmp_path):
        # use the olive managed python environment as the test environment

        from test.multiple_ep.utils import create_and_run_workflow

        system_config = SystemConfig(
            type="PythonEnvironment",
            config=PythonEnvironmentTargetUserConfig(
                accelerators=[
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"]}
                ],
                olive_managed_env=True,
            ),
        )
        cpu_res, openvino_res = create_and_run_workflow(
            tmp_path, system_config, self.input_model_config, get_custom_metric()
        )
        assert cpu_res.metrics.value.__root__
        assert openvino_res.metrics.value.__root__
