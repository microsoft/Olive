import platform
from test.integ_test.evaluator.docker_eval.utils import get_docker_target, get_onnx_model

import pytest

from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR
from olive.model.config.model_config import ModelConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.perf_tuning import OrtPerfTuning


@pytest.mark.skipif(platform.system() == "Windows", reason="Docker target does not support windows")
def test_evaluate_model(tmp_path):
    docker_target = get_docker_target()
    model_config = get_onnx_model()
    model_conf = ModelConfig.parse_obj({"type": "ONNXModel", "config": model_config})

    p = create_pass_from_dict(OrtPerfTuning, {}, True, DEFAULT_CPU_ACCELERATOR)
    output_model = docker_target.run_pass(p, model_conf, None, tmp_path)
    assert output_model.config["inference_settings"]["execution_provider"] == DEFAULT_CPU_ACCELERATOR.execution_provider
