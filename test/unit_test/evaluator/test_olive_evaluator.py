# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from test.unit_test.utils import (
    get_accuracy_metric,
    get_custom_metric,
    get_custom_metric_no_eval,
    get_latency_metric,
    get_mock_openvino_model,
    get_mock_snpe_model,
    get_onnx_model,
    get_pytorch_model,
    get_throughput_metric,
)
from types import FunctionType
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest

from olive.common.pydantic_v1 import ValidationError
from olive.evaluator.metric import AccuracySubType, LatencySubType, ThroughputSubType
from olive.evaluator.olive_evaluator import (
    OliveEvaluator,
    OliveEvaluatorConfig,
    OnnxEvaluator,
    OpenVINOEvaluator,
    PyTorchEvaluator,
    SNPEEvaluator,
)
from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, Device
from olive.systems.local import LocalSystem


class TestOliveEvaluator:
    ACCURACY_TEST_CASE: ClassVar[list] = [
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_accuracy_metric(AccuracySubType.RECALL),
            "olive.evaluator.accuracy.Recall",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_accuracy_metric(AccuracySubType.AUROC),
            "olive.evaluator.accuracy.AUROC",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            get_accuracy_metric(AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            get_accuracy_metric(AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            get_accuracy_metric(AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            get_accuracy_metric(AccuracySubType.RECALL),
            "olive.evaluator.accuracy.Recall",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            get_accuracy_metric(AccuracySubType.AUROC),
            "olive.evaluator.accuracy.AUROC",
            0.99,
        ),
    ]

    @pytest.mark.parametrize(
        "evaluator,model_loader,metric,acc_subtype,expected_res",
        ACCURACY_TEST_CASE,
    )
    def test_evaluate_accuracy(self, evaluator, model_loader, metric, acc_subtype, expected_res):
        # setup
        with patch(f"{acc_subtype}.measure") as mock_acc:
            mock_acc.return_value = expected_res

            olive_model = model_loader()
            # execute
            actual_res = evaluator.evaluate(olive_model, None, [metric])

            # assert
            mock_acc.assert_called_once()
            for sub_type in metric.sub_types:
                assert expected_res == actual_res.get_value(metric.name, sub_type.name)

    LATENCY_TEST_CASE: ClassVar[list] = [
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.AVG, LatencySubType.MAX), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.MAX), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.MIN), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P50), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P75), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P90), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P95), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P99), 10),
        (PyTorchEvaluator(), get_pytorch_model, get_latency_metric(LatencySubType.P999), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.AVG), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.MAX), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.MIN), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P50), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P75), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P90), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P95), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P99), 10),
        (OnnxEvaluator(), get_onnx_model, get_latency_metric(LatencySubType.P999), 10),
    ]

    @pytest.mark.parametrize(
        "evaluator,model_loader,metric,expected_res",
        LATENCY_TEST_CASE,
    )
    def test_evaluate_latency(self, evaluator, model_loader, metric, expected_res):
        olive_model = model_loader()
        # execute
        actual_res = evaluator.evaluate(olive_model, None, [metric])

        # assert
        for sub_type in metric.sub_types:
            assert expected_res > actual_res.get_value(metric.name, sub_type.name)

    @pytest.mark.parametrize(
        "execution_providers",
        [
            ("CPUExecutionProvider", {}),
            "CPUExecutionProvider",
            ["CPUExecutionProvider"],
            [("CPUExecutionProvider", {})],
        ],
    )
    def test_evaluate_latency_with_eps(self, execution_providers):
        model = get_onnx_model()
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        evaluator.evaluate(model, None, [latency_metric], Device.CPU, execution_providers)

    @pytest.mark.parametrize(
        "execution_providers,exception_type",
        [(("CPUExecutionProvider", {}, {}), ValueError), (("CPUExecutionProvider",), ValueError)],
    )
    def test_evaluate_latency_with_wrong_ep(self, execution_providers, exception_type):
        model = get_onnx_model()
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        with pytest.raises(exception_type):
            evaluator.evaluate(model, None, [latency_metric], Device.CPU, execution_providers)

    @pytest.mark.parametrize(
        "execution_providers",
        [
            ["CPUExecutionProvider", "OpenVINOExecutionProvider"],
            [("CPUExecutionProvider", {}), ("OpenVINOExecutionProvider", {})],
            [("CPUExecutionProvider", {}), "OpenVINOExecutionProvider"],
            ["CPUExecutionProvider", ("OpenVINOExecutionProvider", {})],
        ],
    )
    @patch("onnxruntime.get_available_providers")
    def test_evaluate_latency_with_unsupported_ep(self, mock_get_available_providers, execution_providers):
        mock_get_available_providers.return_value = ["CPUExecutionProvider"]
        model = get_onnx_model()
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        with pytest.raises(
            OliveEvaluationError,
            match="The onnxruntime fallback happens. OpenVINOExecutionProvider is not in the session providers",
        ):
            evaluator.evaluate(model, None, [latency_metric], Device.CPU, execution_providers)

    THROUGHPUT_TEST_CASE: ClassVar[list] = [
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            get_throughput_metric(ThroughputSubType.AVG, ThroughputSubType.MAX),
            1e8,
        ),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.MAX), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.MIN), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P50), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P75), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P90), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P95), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P99), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, get_throughput_metric(ThroughputSubType.P999), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.AVG), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.MAX), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.MIN), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P50), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P75), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P90), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P95), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P99), 1e8),
        (OnnxEvaluator(), get_onnx_model, get_throughput_metric(ThroughputSubType.P999), 1e8),
    ]

    @pytest.mark.parametrize(
        "evaluator,model_loader,metric,expected_res",
        THROUGHPUT_TEST_CASE,
    )
    def test_evaluate_throughput(self, evaluator, model_loader, metric, expected_res):
        olive_model = model_loader()
        # execute
        actual_res = evaluator.evaluate(olive_model, None, [metric])

        # assert
        for sub_type in metric.sub_types:
            assert expected_res > actual_res.get_value(metric.name, sub_type.name)

    CUSTOM_TEST_CASE: ClassVar[list] = [
        (PyTorchEvaluator(), get_pytorch_model, get_custom_metric(), 0.382715310),
        (OnnxEvaluator(), get_onnx_model, get_custom_metric(), 0.382715310),
        (SNPEEvaluator(), get_mock_snpe_model, get_custom_metric(), 0.382715310),
        (OpenVINOEvaluator(), get_mock_openvino_model, get_custom_metric(), 0.382715310),
    ]

    @pytest.mark.parametrize(
        "evaluator,model_loader,metric,expected_res",
        CUSTOM_TEST_CASE,
    )
    def test_evaluate_custom(self, evaluator, model_loader, metric, expected_res):
        olive_model = model_loader()
        # execute
        actual_res = evaluator.evaluate(olive_model, None, [metric])

        # assert
        for sub_type in metric.sub_types:
            assert actual_res.get_value(metric.name, sub_type.name) == expected_res

    def test_evaluate_custom_no_eval(self):
        evaluator = PyTorchEvaluator()
        olive_model = get_pytorch_model()
        metric = get_custom_metric_no_eval()
        with pytest.raises(ValueError, match="evaluate_func or metric_func is not specified in the metric config"):
            evaluator.evaluate(olive_model, None, [metric])

    @pytest.mark.parametrize(
        "dataloader_func_kwargs", [None, {"kwarg_1": "value_1"}, {"kwarg_1": "value_1", "kwarg_2": "value_2"}]
    )
    def test_dataloader_func_kwargs(self, dataloader_func_kwargs):
        # setup
        dataloader_func = MagicMock(spec=FunctionType)
        data_dir = None
        batch_size = 1
        model_framework = "PyTorch"
        user_config = {"dataloader_func": dataloader_func, "batch_size": batch_size, "data_dir": data_dir}
        if dataloader_func_kwargs:
            user_config["func_kwargs"] = {"dataloader_func": dataloader_func_kwargs}
        metric = get_latency_metric(LatencySubType.AVG, user_config=user_config)

        # execute
        OliveEvaluator.get_user_config(model_framework, None, metric)

        # assert
        dataloader_func.assert_called_once_with(
            data_dir, batch_size, model_framework=model_framework, **(dataloader_func_kwargs or {})
        )

    # this is enough to test the kwargs for `evaluate_func`, `metric_func` and `post_process_func`
    # since they are all using the same `get_user_config` method
    @pytest.mark.parametrize(
        "evaluate_func_kwargs", [None, {"kwarg_1": "value_1"}, {"kwarg_1": "value_1", "kwarg_2": "value_2"}]
    )
    def test_evaluate_func_kwargs(self, evaluate_func_kwargs):
        # setup
        dataloader_func = MagicMock(spec=FunctionType)
        evaluate_func = MagicMock(spec=FunctionType)
        user_config = {"dataloader_func": dataloader_func, "evaluate_func": evaluate_func}
        if evaluate_func_kwargs:
            user_config["func_kwargs"] = {"evaluate_func": evaluate_func_kwargs}
        metric = get_custom_metric(user_config=user_config)

        # execute
        _, eval_func, _ = OliveEvaluator.get_user_config(None, None, metric)
        eval_func("model", "data_dir", "batch_size", "device", "execution_providers")

        # assert
        evaluate_func.assert_called_once_with(
            "model", "data_dir", "batch_size", "device", "execution_providers", **(evaluate_func_kwargs or {})
        )


@pytest.mark.skip(reason="Requires custom onnxruntime build with mpi enabled")
class TestDistributedOnnxEvaluator:
    def test_evaluate(self):
        from olive.model import DistributedOnnxModel

        model = DistributedOnnxModel("examples/switch", "model_4n_2l_8e_{:02d}.onnx", 2)

        user_config = {
            "user_script": "examples/switch/user_script.py",
            "dataloader_func": "create_dataloader",
            "batch_size": 1,
        }
        # accuracy_metric = get_accuracy_metric(AccuracySubType.ACCURACY_SCORE, user_config=user_config)
        latency_metric = get_latency_metric(LatencySubType.AVG, user_config=user_config)
        # metrics = [accuracy_metric, latency_metric]
        metrics = [latency_metric]

        target = LocalSystem()

        # execute
        actual_res = target.evaluate_model(model, None, metrics, DEFAULT_CPU_ACCELERATOR)

        # assert
        for sub_type in latency_metric.sub_types:
            assert actual_res.get_value(latency_metric.name, sub_type.name) > 1


class TestOliveEvaluatorConfig:
    @pytest.mark.parametrize(
        "priorities, has_exception", [((1, 2), False), ((1, 1), True), ((2, 1), False), ((1, 0), True)]
    )
    def test_priority(self, priorities, has_exception):
        metric_config = [
            {
                "name": "latency1",
                "type": "latency",
                "sub_types": [
                    {
                        "name": "avg",
                        "priority": priorities[0],
                        "goal": {"type": "percent-min-improvement", "value": 20},
                    },
                    {"name": "max"},
                    {"name": "min"},
                ],
            },
            {
                "name": "latency2",
                "type": "latency",
                "sub_types": [
                    {
                        "name": "avg",
                        "priority": priorities[1],
                        "goal": {"type": "percent-min-improvement", "value": 20},
                    },
                    {"name": "max"},
                    {"name": "min"},
                ],
            },
        ]
        if has_exception:
            with pytest.raises(ValidationError):
                OliveEvaluatorConfig(metrics=metric_config)
        else:
            OliveEvaluatorConfig(metrics=metric_config)
