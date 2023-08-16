# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path
from test.integ_test.evaluator.azureml_eval.utils import get_latency_metric
from test.unit_test.utils import create_onnx_model_file, get_onnx_model

import pytest

from olive.engine import Engine
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.passes.onnx import OrtPerfTuning
from olive.systems.docker.docker_system import DockerSystem


class TestOliveManagedDockerSystem:
    @pytest.fixture(autouse=True)
    def setup(self):
        # use the olive managed Docker system as the test environment
        self.system = DockerSystem(accelerators=["cpu"], olive_managed_env=True)
        self.execution_providers = ["CPUExecutionProvider", "OpenVINOExecutionProvider"]
        create_onnx_model_file()
        self.input_model = get_onnx_model()

    def test_run_pass_evaluate(self):
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

        metric = get_latency_metric()
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        options = {"execution_providers": self.execution_providers}
        engine = Engine(options, target=self.system, evaluator_config=evaluator_config)
        engine.register(OrtPerfTuning)
        engine.run(self.input_model, output_dir=output_dir)
