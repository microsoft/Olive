# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.integ_test.utils import get_olive_workspace_config
from test.multiple_ep.utils import download_data, download_models, get_latency_metric, get_onnx_model

import pytest

from olive.azureml.azureml_client import AzureMLClientConfig
from olive.engine import Engine
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes.onnx import OrtPerfTuning
from olive.systems.azureml.aml_system import AzureMLSystem


class TestOliveAzureMLSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed AzureML system as the test environment
        aml_compute = "cpu-cluster"
        azureml_client_config = AzureMLClientConfig(**get_olive_workspace_config())

        self.system = AzureMLSystem(
            azureml_client_config=azureml_client_config,
            aml_compute=aml_compute,
            accelerators=["cpu"],
            olive_managed_env=True,
            requirements_file=Path(__file__).parent / "requirements.txt",
            is_dev=True,
        )

        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        download_models()
        self.input_model = ONNXModel(model_path=get_onnx_model())
        download_data()

    def test_run_pass_evaluate(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name

        metric = get_latency_metric()
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {"execution_providers": self.execution_providers}
        engine = Engine(options, target=self.system, host=self.system, evaluator_config=evaluator_config)
        engine.register(OrtPerfTuning)
        output = engine.run(self.input_model, output_dir=output_dir)
        cpu_res = output[AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")]
        openvino_res = output[
            AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="OpenVINOExecutionProvider")
        ]
        assert cpu_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
        assert openvino_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
