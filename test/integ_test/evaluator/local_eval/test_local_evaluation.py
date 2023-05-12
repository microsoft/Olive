# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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
    openvino_post_process,
    post_process,
)

import pytest

from olive.model import ONNXModel, OpenVINOModel, PyTorchModel
from olive.systems.local import LocalSystem


class TestLocalEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        yield
        delete_directories()

    EVALUATION_TEST_CASE = [
        (PyTorchModel, get_pytorch_model(), get_accuracy_metric(post_process), 0.99),
        (PyTorchModel, get_pytorch_model(), get_latency_metric(), 0.001),
        (PyTorchModel, get_huggingface_model(), get_hf_accuracy_metric(), 0.1),
        (PyTorchModel, get_huggingface_model(), get_hf_latency_metric(), 0.001),
        (ONNXModel, get_onnx_model(), get_accuracy_metric(post_process), 0.99),
        (ONNXModel, get_onnx_model(), get_latency_metric(), 0.001),
        (OpenVINOModel, get_openvino_model(), get_accuracy_metric(openvino_post_process), 0.99),
        (OpenVINOModel, get_openvino_model(), get_latency_metric(), 0.001),
    ]

    @pytest.mark.parametrize(
        "model_cls,model_config,metric,expected_res",
        EVALUATION_TEST_CASE,
    )
    def test_evaluate_model(self, model_cls, model_config, metric, expected_res):
        olive_model = model_cls(**model_config)
        actual_res = LocalSystem().evaluate_model(olive_model, [metric])[metric.name]
        assert actual_res >= expected_res
