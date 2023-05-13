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

from olive.model import ModelStorageKind, ONNXModel, PyTorchModel


class TestAMLEvaluation:
    @pytest.fixture(scope="class", autouse=True)
    def setup(self):
        get_directories()
        download_models()
        download_data()
        yield
        delete_directories()

    EVALUATION_TEST_CASE = [
        (PyTorchModel, get_pytorch_model(), get_accuracy_metric(), 0.99),
        (PyTorchModel, get_pytorch_model(), get_latency_metric(), 0.001),
        (ONNXModel, get_onnx_model(), get_accuracy_metric(), 0.99),
        (ONNXModel, get_onnx_model(), get_latency_metric(), 0.001),
    ]

    @pytest.mark.parametrize(
        "model_cls,model_path,metric,expected_res",
        EVALUATION_TEST_CASE,
    )
    def test_evaluate_model(self, model_cls, model_path, metric, expected_res):
        aml_target = get_aml_target()
        olive_model = model_cls(model_path=model_path, model_storage_kind=ModelStorageKind.LocalFile)
        actual_res = aml_target.evaluate_model(olive_model, [metric])[metric.name]
        assert actual_res >= expected_res
