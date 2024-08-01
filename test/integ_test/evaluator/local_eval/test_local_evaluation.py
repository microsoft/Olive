# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from functools import partial
from test.integ_test.evaluator.local_eval.utils import (
    delete_directories,
    get_accuracy_metric,
    get_directories,
    get_hf_accuracy_metric,
    get_hf_latency_metric,
    get_huggingface_model,
    get_latency_metric,
    get_onnx_model,
    get_openvino_model,
    get_pytorch_model,
)
from typing import ClassVar, List

import pytest

from olive.evaluator.metric_result import joint_metric_key
from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ModelConfig
from olive.systems.local import LocalSystem

# pylint: disable=redefined-builtin


class TestLocalEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        yield
        delete_directories()

    EVALUATION_TEST_CASE: ClassVar[List] = [
        ("PyTorchModel", get_pytorch_model, partial(get_accuracy_metric, "mnist_post_process_for_local_eval"), 0.99),
        ("PyTorchModel", get_pytorch_model, get_latency_metric, 0.001),
        ("HfModel", get_huggingface_model, get_hf_accuracy_metric, 0.1),
        ("HfModel", get_huggingface_model, get_hf_latency_metric, 0.001),
        ("ONNXModel", get_onnx_model, partial(get_accuracy_metric, "mnist_post_process_for_local_eval"), 0.99),
        ("ONNXModel", get_onnx_model, get_latency_metric, 0.001),
        (
            "OpenVINOModel",
            get_openvino_model,
            partial(get_accuracy_metric, "mnist_post_process_openvino_for_local_eval"),
            0.99,
        ),
        ("OpenVINOModel", get_openvino_model, get_latency_metric, 0.001),
    ]

    @pytest.mark.parametrize(
        ("type", "model_config_func", "metric_func", "expected_res"),
        EVALUATION_TEST_CASE,
    )
    def test_evaluate_model(self, type, model_config_func, metric_func, expected_res):  # noqa: A002
        model_config = model_config_func()
        metric = metric_func()
        model_config = ModelConfig.parse_obj({"type": type, "config": model_config})
        evaluator_config = OliveEvaluatorConfig(metrics=[metric])
        actual_res = LocalSystem().evaluate_model(model_config, evaluator_config, DEFAULT_CPU_ACCELERATOR)
        for sub_type in metric.sub_types:
            joint_key = joint_metric_key(metric.name, sub_type.name)
            assert actual_res[joint_key].value >= expected_res
