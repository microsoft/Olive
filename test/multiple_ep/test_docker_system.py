# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
import tempfile
from test.integ_test.multiple_ep.utils import download_data, download_models, get_latency_metric, get_onnx_model

import pytest

from olive.engine import Engine
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes.onnx import OrtPerfTuning
from olive.systems.docker.docker_system import DockerSystem


class TestOliveManagedDockerSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed Docker system as the test environment
        self.system = DockerSystem(accelerators=["cpu"], olive_managed_env=True, is_dev=True)
        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        download_models()
        self.input_model = ONNXModel(model_path=get_onnx_model())
        download_data()

    @pytest.mark.skipif(platform.system() == "Windows", reason="Docker target does not support windows")
    def test_run_pass_evaluate(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name

        metric = get_latency_metric()
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {"execution_providers": self.execution_providers}
        engine = Engine(options, target=self.system, evaluator_config=evaluator_config)
        engine.register(OrtPerfTuning)
        output = engine.run(self.input_model, output_dir=output_dir)
        cpu_res = output[AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")]
        openvino_res = output[
            AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="OpenVINOExecutionProvider")
        ]
        assert cpu_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
        assert openvino_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
