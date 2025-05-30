# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import platform
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.hardware.accelerator import ExecutionProvider
from olive.logging import set_default_logger_severity
from olive.model import ModelConfig
from test.multiple_ep.utils import get_directories

# pylint: disable=attribute-defined-outside-init


@pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="Docker target does not support windows")
class TestOliveManagedDockerSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        from olive.systems.system_config import DockerTargetUserConfig, SystemConfig
        from test.multiple_ep.utils import download_data, download_models, get_onnx_model

        # use the olive managed Docker system as the test environment
        self.system_config = SystemConfig(
            type="Docker",
            config=DockerTargetUserConfig(
                accelerators=[
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"]}
                ],
                requirements_file=Path(__file__).parent / "requirements.txt",
                olive_managed_env=True,
                is_dev=True,
            ),
        )
        get_directories()
        download_models()
        self.input_model_config = ModelConfig.parse_obj(
            {"type": "ONNXModel", "config": {"model_path": get_onnx_model()}}
        )
        download_data()

    def test_run_pass_evaluate(self, tmp_path, caplog):
        logger = logging.getLogger("olive")
        logger.propagate = True

        from test.multiple_ep.utils import create_and_run_workflow, get_latency_metric

        set_default_logger_severity(0)
        workflow_output = create_and_run_workflow(
            tmp_path,
            self.system_config,
            self.input_model_config,
            get_latency_metric(),
            only_target=True,
        )
        assert workflow_output.cpu[str(ExecutionProvider.CPUExecutionProvider)][0].metrics
        assert workflow_output.cpu[str(ExecutionProvider.OpenVINOExecutionProvider)][0].metrics
        assert "Creating olive_managed_env Docker with EP CPUExecutionProvider" in caplog.text
        assert "Creating olive_managed_env Docker with EP OpenVINOExecutionProvider" in caplog.text
