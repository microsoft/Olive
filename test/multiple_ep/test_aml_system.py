# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from pathlib import Path

import pytest

from olive.common.constants import OS
from olive.model import ModelConfig

# pylint: disable=attribute-defined-outside-init


@pytest.mark.skip(reason="Skip test on AzureML.")
@pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="Skip test on Windows.")
class TestOliveAzureMLSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed AzureML system as the test environment
        from test.integ_test.utils import get_olive_workspace_config
        from test.multiple_ep.utils import download_data, download_models, get_onnx_model

        from olive.azureml.azureml_client import AzureMLClientConfig
        from olive.systems.system_config import AzureMLTargetUserConfig, SystemConfig

        aml_compute = "cpu-cluster"
        azureml_client_config = AzureMLClientConfig(**get_olive_workspace_config())

        self.system_config = SystemConfig(
            type="AzureML",
            config=AzureMLTargetUserConfig(
                azureml_client_config=azureml_client_config,
                aml_compute=aml_compute,
                accelerators=[
                    {"device": "cpu", "execution_providers": ["CPUExecutionProvider", "OpenVINOExecutionProvider"]}
                ],
                olive_managed_env=True,
                requirements_file=Path(__file__).parent / "requirements.txt",
                is_dev=True,
            ),
        )

        download_models()
        self.input_model_config = ModelConfig.parse_obj(
            {"type": "ONNXModel", "config": {"model_path": get_onnx_model()}}
        )
        download_data()

    def test_run_pass_evaluate(self, tmp_path):
        from test.multiple_ep.utils import create_and_run_workflow, get_latency_metric

        cpu_res, openvino_res = create_and_run_workflow(
            tmp_path, self.system_config, self.input_model_config, get_latency_metric()
        )
        assert cpu_res.metrics.value.__root__
        assert openvino_res.metrics.value.__root__
