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
from typing import ClassVar, List

import pytest

from olive.evaluator.metric import joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ModelConfig


class TestDockerEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        download_models()
        download_data()
        yield
        delete_directories()

    EVALUATION_TEST_CASE: ClassVar[List] = [
        ("PyTorchModel", get_pytorch_model(), get_accuracy_metric("post_process"), 0.99),
        ("PyTorchModel", get_pytorch_model(), get_latency_metric(), 0.001),
        ("PyTorchModel", get_huggingface_model(), get_accuracy_metric("hf_post_process", "create_hf_dataloader"), 0.1),
        ("PyTorchModel", get_huggingface_model(), get_latency_metric("create_hf_dataloader"), 0.001),
        ("ONNXModel", get_onnx_model(), get_accuracy_metric("post_process"), 0.99),
        ("ONNXModel", get_onnx_model(), get_latency_metric(), 0.001),
        ("OpenVINOModel", get_openvino_model(), get_accuracy_metric("openvino_post_process"), 0.99),
        ("OpenVINOModel", get_openvino_model(), get_latency_metric(), 0.001),
    ]

    @pytest.mark.parametrize(
        "model_type,model_config,metric,expected_res",
        EVALUATION_TEST_CASE,
    )
    @pytest.mark.skipif(platform.system() == "Windows", reason="Docker target does not support windows")
    def test_evaluate_model(self, model_type, model_config, metric, expected_res):
        docker_target = get_docker_target()
        model_conf = ModelConfig.parse_obj({"type": model_type, "config": model_config})
        actual_res = docker_target.evaluate_model(model_conf, None, [metric], DEFAULT_CPU_ACCELERATOR)
        for sub_type in metric.sub_types:
            joint_key = joint_metric_key(metric.name, sub_type.name)
            assert actual_res[joint_key].value >= expected_res
