# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
import tempfile

import pytest

from olive.engine import Engine
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig
from olive.passes.onnx import OrtPerfTuning


@pytest.mark.skipif(platform.system() == "Windows", reason="Docker target does not support windows")
class TestOliveManagedDockerSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        from test.multiple_ep.utils import download_data, download_models, get_onnx_model

        from olive.systems.docker.docker_system import DockerSystem

        # use the olive managed Docker system as the test environment
        self.system = DockerSystem(accelerators=["cpu"], olive_managed_env=True, is_dev=True)
        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        download_models()
        self.input_model_config = ModelConfig.parse_obj(
            {"type": "ONNXModel", "config": {"model_path": get_onnx_model()}}
        )
        download_data()

    def test_run_pass_evaluate(self):
        from test.multiple_ep.utils import get_latency_metric

        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name

        metric = get_latency_metric()
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {"execution_providers": self.execution_providers}
        engine = Engine(options, target=self.system, evaluator_config=evaluator_config)
        engine.register(OrtPerfTuning)
        output = engine.run(self.input_model_config, output_dir=output_dir)
        cpu_res = output[AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")]
        openvino_res = output[
            AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="OpenVINOExecutionProvider")
        ]
        assert cpu_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
        assert openvino_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
