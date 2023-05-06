# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from test.integ_test.evaluator.docker_eval.utils import (
    delete_directories,
    download_data,
    download_models,
    get_accuracy_metric,
    get_directories,
    get_docker_target,
    get_huggingface_model,
    get_latency_metric,
    get_onnx_model,
    get_openvino_model,
    get_pytorch_model,
)

import pytest

from olive.evaluator.olive_evaluator import OliveEvaluator
from olive.model import ONNXModel, OpenVINOModel, PyTorchModel


class TestDockerEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        download_models()
        download_data()
        yield
        delete_directories()

    EVALUATION_TEST_CASE = [
        (PyTorchModel, get_pytorch_model(), get_accuracy_metric("post_process"), 0.99),
        (PyTorchModel, get_pytorch_model(), get_latency_metric(), 0.001),
        (PyTorchModel, get_huggingface_model(), get_accuracy_metric("hf_post_process", "create_hf_dataloader"), 0.1),
        (PyTorchModel, get_huggingface_model(), get_latency_metric("create_hf_dataloader"), 0.001),
        (ONNXModel, get_onnx_model(), get_accuracy_metric("post_process"), 0.99),
        (ONNXModel, get_onnx_model(), get_latency_metric(), 0.001),
        (OpenVINOModel, get_openvino_model(), get_accuracy_metric("openvino_post_process"), 0.99),
        (OpenVINOModel, get_openvino_model(), get_latency_metric(), 0.001),
    ]

    @pytest.mark.parametrize(
        "model_cls,model_config,metric,expected_res",
        EVALUATION_TEST_CASE,
    )
    @pytest.mark.skipif(platform.system() == "Windows", reason="Docker target does not support windows")
    def test_evaluate_model(self, model_cls, model_config, metric, expected_res):
        docker_target = get_docker_target()
        olive_model = model_cls(**model_config)
        evaluator = OliveEvaluator(metrics=[metric])
        actual_res = evaluator.evaluate(olive_model, docker_target)[metric.name]
        assert actual_res >= expected_res
