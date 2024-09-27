import platform
from test.integ_test.evaluator.docker_eval.utils import (
    delete_directories,
    download_models,
    get_directories,
    get_docker_target,
    get_onnx_model,
)

import pytest

from olive.common.constants import OS
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR
from olive.logging import set_default_logger_severity
from olive.model.config.model_config import ModelConfig
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.session_params_tuning import OrtSessionParamsTuning


@pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="Docker target does not support windows")
@pytest.fixture(scope="module", autouse=True)
def setup():
    get_directories()
    download_models()
    yield
    delete_directories()


@pytest.mark.skipif(platform.system() == OS.WINDOWS, reason="Docker target does not support windows")
def test_pass_runner(tmp_path):
    docker_target = get_docker_target()
    model_config = get_onnx_model()
    model_conf = ModelConfig.parse_obj({"type": "ONNXModel", "config": model_config})

    set_default_logger_severity(0)
    p = create_pass_from_dict(OrtSessionParamsTuning, {}, True, DEFAULT_CPU_ACCELERATOR)
    output_model = docker_target.run_pass(p, model_conf, tmp_path)
    result_eps = output_model.config["inference_settings"]["execution_provider"]
    assert result_eps == [DEFAULT_CPU_ACCELERATOR.execution_provider]
    assert output_model.config["model_path"] == model_config["model_path"]
