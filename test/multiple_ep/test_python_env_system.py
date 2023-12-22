# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from test.unit_test.utils import create_onnx_model_file, get_latency_metric, get_onnx_model_config

import pytest

from olive.engine import Engine
from olive.evaluator.metric import LatencySubType
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import Device
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec, create_accelerators
from olive.passes.onnx import OrtPerfTuning
from olive.systems.python_environment import PythonEnvironmentSystem

# pylint: disable=attribute-defined-outside-init


class TestOliveManagedPythonEnvironmentSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        create_onnx_model_file()
        self.input_model_config = get_onnx_model_config()

    @pytest.mark.skip(reason="No machine to test DML execution provider")
    def test_run_pass_evaluate_windows(self, tmpdir):
        # use the olive managed python environment as the test environment
        self.system = PythonEnvironmentSystem(
            accelerators=["gpu"],
            olive_managed_env=True,
        )
        self.execution_providers = ["DmlExecutionProvider", "OpenVINOExecutionProvider"]
        output_dir = tmpdir

        metric = get_latency_metric(LatencySubType.AVG)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        engine = Engine(target=self.system, host=self.system, evaluator_config=evaluator_config)
        accelerator_specs = create_accelerators(self.system, self.execution_providers)

        engine.register(OrtPerfTuning)
        output = engine.run(
            self.input_model_config, accelerator_specs, output_dir=output_dir, evaluate_input_model=True
        )
        dml_res = output[AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="DmlExecutionProvider")]
        openvino_res = output[
            AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="OpenVINOExecutionProvider")
        ]
        assert dml_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]
        assert openvino_res[tuple(engine.pass_flows[0])]["metrics"]["latency-avg"]

    @pytest.mark.skipif(platform.system() == "Windows", reason="Test for Linux only")
    def test_run_pass_evaluate_linux(self, tmpdir):
        # use the olive managed python environment as the test environment
        self.system = PythonEnvironmentSystem(
            accelerators=["cpu"],
            olive_managed_env=True,
        )
        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        output_dir = tmpdir

        metric = get_latency_metric(LatencySubType.AVG)
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        engine = Engine(target=self.system, host=self.system, evaluator_config=evaluator_config)
        accelerator_specs = create_accelerators(self.system, self.execution_providers)
        engine.register(OrtPerfTuning)
        output = engine.run(
            self.input_model_config, accelerator_specs, output_dir=output_dir, evaluate_input_model=True
        )
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
