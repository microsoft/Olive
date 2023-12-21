# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from pathlib import Path

import pytest

from olive.engine import Engine
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec, create_accelerators
from olive.model import ModelConfig
from olive.passes.onnx import OrtPerfTuning

# pylint: disable=attribute-defined-outside-init


@pytest.mark.skipif(platform.system() == "Windows", reason="Skip test on Windows.")
class TestOliveAzureMLSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed AzureML system as the test environment
        from test.integ_test.utils import get_olive_workspace_config
        from test.multiple_ep.utils import download_data, download_models, get_onnx_model

        from olive.azureml.azureml_client import AzureMLClientConfig
        from olive.systems.azureml.aml_system import AzureMLSystem

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
        self.input_model_config = ModelConfig.parse_obj(
            {"type": "ONNXModel", "config": {"model_path": get_onnx_model()}}
        )
        download_data()

    def test_run_pass_evaluate(self, tmpdir):
        from test.multiple_ep.utils import get_latency_metric

        output_dir = tmpdir

        metric = get_latency_metric()
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        engine = Engine(target=self.system, host=self.system, evaluator_config=evaluator_config)
        accelerator_specs = create_accelerators(self.system, self.execution_providers)
        engine.register(OrtPerfTuning)
        output = engine.run(self.input_model_config, accelerator_specs, output_dir=output_dir)
        cpu_res = next(iter(output[DEFAULT_CPU_ACCELERATOR].nodes.values()))
        openvino_res = next(
            iter(
                output[
                    AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="OpenVINOExecutionProvider")
                ].nodes.values()
            )
        )
        assert cpu_res.metrics.value.__root__
        assert openvino_res.metrics.value.__root__
