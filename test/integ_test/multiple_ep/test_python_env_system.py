# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
import tempfile
from test.unit_test.utils import create_onnx_model_file, get_latency_metric, get_onnx_model

import pytest

from olive.engine import Engine
from olive.evaluator.metric import LatencySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.passes.onnx import OrtPerfTuning
from olive.systems.python_environment import PythonEnvironmentSystem


class TestOliveManagedPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed python environment as the test environment
        self.system = PythonEnvironmentSystem(accelerators=["cpu"], olive_managed_env=True)
        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        create_onnx_model_file()
        self.input_model = get_onnx_model()

    @pytest.mark.skipif(platform.system() == "Windows", reason="OpenVINO does not support windows")
    def test_run_pass_evaluate(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = temp_dir.name

        metric = get_latency_metric(LatencySubType.AVG)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {"execution_providers": self.execution_providers}
        engine = Engine(options, target=self.system, host=self.system, evaluator_config=evaluator_config)
        engine.register(OrtPerfTuning)
        output = engine.run(self.input_model, output_dir=output_dir, evaluate_input_model=True)
        cpu_res = output[AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="CPUExecutionProvider")]
        openvino_res = output[
            AcceleratorSpec(accelerator_type=Device.CPU, execution_provider="OpenVINOExecutionProvider")
        ]
        assert cpu_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
        assert openvino_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
