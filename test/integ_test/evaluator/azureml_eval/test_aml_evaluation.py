# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.integ_test.evaluator.azureml_eval.utils import (
    delete_directories,
    download_data,
    download_models,
    get_accuracy_metric,
    get_aml_target,
    get_directories,
    get_latency_metric,
    get_onnx_model,
    get_pytorch_model,
)

import pytest

from olive.evaluator.metric import joint_metric_key
from olive.hardware import DEFAULT_CPU_ACCELERATOR
from olive.model import ModelConfig


class TestAMLEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        download_models()
        download_data()
        yield
        delete_directories()

    EVALUATION_TEST_CASE = [
        ("PyTorchModel", get_pytorch_model(), get_accuracy_metric(), 0.99),
        ("PyTorchModel", get_pytorch_model(), get_latency_metric(), 0.001),
        ("ONNXModel", get_onnx_model(), get_accuracy_metric(), 0.99),
        ("ONNXModel", get_onnx_model(), get_latency_metric(), 0.001),
    ]

    @pytest.mark.parametrize(
        "model_type,model_path,metric,expected_res",
        EVALUATION_TEST_CASE,
    )
    def test_evaluate_model(self, model_type, model_path, metric, expected_res):
        aml_target = get_aml_target()
        config = ModelConfig.parse_obj({"type": model_type, "config": {"model_path": model_path}})
        actual_res = aml_target.evaluate_model(config, None, [metric], DEFAULT_CPU_ACCELERATOR)
        for sub_type in metric.sub_types:
            joint_key = joint_metric_key(metric.name, sub_type.name)
            assert actual_res[joint_key].value >= expected_res
