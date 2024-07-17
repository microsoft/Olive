# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from functools import partial
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
from typing import ClassVar, List
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
from olive.hardware.accelerator import Device


class TestOliveEvaluator:
    ACCURACY_TEST_CASE: ClassVar[List] = [
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_accuracy_metric, AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_accuracy_metric, AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_accuracy_metric, AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_accuracy_metric, AccuracySubType.RECALL),
            "olive.evaluator.accuracy.Recall",
            0.99,
        ),
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_accuracy_metric, AccuracySubType.AUROC),
            "olive.evaluator.accuracy.AUROC",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            partial(get_accuracy_metric, AccuracySubType.ACCURACY_SCORE),
            "olive.evaluator.accuracy.AccuracyScore",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            partial(get_accuracy_metric, AccuracySubType.F1_SCORE),
            "olive.evaluator.accuracy.F1Score",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            partial(get_accuracy_metric, AccuracySubType.PRECISION),
            "olive.evaluator.accuracy.Precision",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            partial(get_accuracy_metric, AccuracySubType.RECALL),
            "olive.evaluator.accuracy.Recall",
            0.99,
        ),
        (
            OnnxEvaluator(),
            get_onnx_model,
            partial(get_accuracy_metric, AccuracySubType.AUROC),
            "olive.evaluator.accuracy.AUROC",
            0.99,
        ),
    ]

    @pytest.mark.parametrize(
        ("evaluator", "model_loader", "metric_func", "acc_subtype", "expected_res"),
        ACCURACY_TEST_CASE,
    )
    def test_evaluate_accuracy(self, evaluator, model_loader, metric_func, acc_subtype, expected_res):
        # setup
        with patch(f"{acc_subtype}.measure") as mock_acc:
            mock_acc.return_value = expected_res

            olive_model = model_loader()
            metric = metric_func()
            # execute
            actual_res = evaluator.evaluate(olive_model, [metric])

            # assert
            mock_acc.assert_called_once()
            for sub_type in metric.sub_types:
                assert expected_res == actual_res.get_value(metric.name, sub_type.name)

    LATENCY_TEST_CASE: ClassVar[List] = [
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_latency_metric, LatencySubType.AVG, LatencySubType.MAX),
            10,
        ),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.MAX), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.MIN), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P50), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P75), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P90), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P95), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P99), 10),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_latency_metric, LatencySubType.P999), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.AVG), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.MAX), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.MIN), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P50), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P75), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P90), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P95), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P99), 10),
        (OnnxEvaluator(), get_onnx_model, partial(get_latency_metric, LatencySubType.P999), 10),
    ]

    @pytest.mark.parametrize(
        ("evaluator", "model_loader", "metric_func", "expected_res"),
        LATENCY_TEST_CASE,
    )
    def test_evaluate_latency(self, evaluator, model_loader, metric_func, expected_res):
        olive_model = model_loader()
        # execute
        metric = metric_func()
        actual_res = evaluator.evaluate(olive_model, [metric])

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
        evaluator.evaluate(model, [latency_metric], Device.CPU, execution_providers)

    @pytest.mark.parametrize(
        ("execution_providers", "exception_type"),
        [(("CPUExecutionProvider", {}, {}), ValueError), (("CPUExecutionProvider",), ValueError)],
    )
    def test_evaluate_latency_with_wrong_ep(self, execution_providers, exception_type):
        model = get_onnx_model()
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        with pytest.raises(exception_type):
            evaluator.evaluate(model, [latency_metric], Device.CPU, execution_providers)

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
    def test_evaluate_latency_with_unsupported_ep(self, get_available_providers_mock, execution_providers):
        get_available_providers_mock.return_value = ["CPUExecutionProvider"]
        model = get_onnx_model()
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        with pytest.raises(
            OliveEvaluationError,
            match="The onnxruntime fallback happens. OpenVINOExecutionProvider is not in the session providers",
        ):
            evaluator.evaluate(model, [latency_metric], Device.CPU, execution_providers)

    THROUGHPUT_TEST_CASE: ClassVar[List] = [
        (
            PyTorchEvaluator(),
            get_pytorch_model,
            partial(get_throughput_metric, ThroughputSubType.AVG, ThroughputSubType.MAX),
            1e8,
        ),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.MAX), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.MIN), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P50), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P75), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P90), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P95), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P99), 1e8),
        (PyTorchEvaluator(), get_pytorch_model, partial(get_throughput_metric, ThroughputSubType.P999), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.AVG), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.MAX), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.MIN), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P50), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P75), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P90), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P95), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P99), 1e8),
        (OnnxEvaluator(), get_onnx_model, partial(get_throughput_metric, ThroughputSubType.P999), 1e8),
    ]

    @pytest.mark.parametrize(
        ("evaluator", "model_loader", "metric_func", "expected_res"),
        THROUGHPUT_TEST_CASE,
    )
    def test_evaluate_throughput(self, evaluator, model_loader, metric_func, expected_res):
        olive_model = model_loader()
        # execute
        metric = metric_func()
        actual_res = evaluator.evaluate(olive_model, [metric])

        # assert
        for sub_type in metric.sub_types:
            assert expected_res > actual_res.get_value(metric.name, sub_type.name)

    CUSTOM_TEST_CASE: ClassVar[List] = [
        (PyTorchEvaluator(), get_pytorch_model, get_custom_metric, 0.382715310),
        (OnnxEvaluator(), get_onnx_model, get_custom_metric, 0.382715310),
        (SNPEEvaluator(), get_mock_snpe_model, get_custom_metric, 0.382715310),
        (OpenVINOEvaluator(), get_mock_openvino_model, get_custom_metric, 0.382715310),
    ]

    @pytest.mark.parametrize(
        ("evaluator", "model_loader", "metric_func", "expected_res"),
        CUSTOM_TEST_CASE,
    )
    def test_evaluate_custom(self, evaluator, model_loader, metric_func, expected_res):
        olive_model = model_loader()
        # execute
        metric = metric_func()
        actual_res = evaluator.evaluate(olive_model, [metric])

        # assert
        for sub_type in metric.sub_types:
            assert actual_res.get_value(metric.name, sub_type.name) == expected_res

    def test_evaluate_custom_no_eval(self):
        evaluator = PyTorchEvaluator()
        olive_model = get_pytorch_model()
        metric = get_custom_metric_no_eval()
        with pytest.raises(ValueError, match="evaluate_func or metric_func is not specified in the metric config"):
            evaluator.evaluate(olive_model, [metric])

    # this is enough to test the kwargs for `evaluate_func`, `metric_func` and `post_process_func`
    # since they are all using the same `get_user_config` method
    @pytest.mark.parametrize(
        "evaluate_func_kwargs", [None, {"kwarg_1": "value_1"}, {"kwarg_1": "value_1", "kwarg_2": "value_2"}]
    )
    def test_evaluate_func_kwargs(self, evaluate_func_kwargs):
        # setup
        evaluate_func = MagicMock(spec=FunctionType)
        user_config = {"evaluate_func": evaluate_func}
        if evaluate_func_kwargs:
            user_config["evaluate_func_kwargs"] = evaluate_func_kwargs
        metric = get_custom_metric(user_config=user_config)

        # execute
        _, eval_func, _ = OliveEvaluator.get_user_config(None, metric)
        eval_func("model", "device", "execution_providers")

        # assert
        evaluate_func.assert_called_once_with("model", "device", "execution_providers", **(evaluate_func_kwargs or {}))

    @patch("onnxruntime.InferenceSession")
    def test_evaluate_latency_with_tunable_op(self, inference_session_mock):
        tuning_result = [
            {
                "ep": "ROCMExecutionProvider",
                "results": {
                    "onnxruntime::rocm::tunable::blas::internal::GemmTunableOp<__half, ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor>": {  # noqa: E501
                        "NN_992_4096_4096": 300,
                        "NN_992_4096_11008": 664,
                        "NN_984_4096_4096": 1295,
                    },
                    "onnxruntime::contrib::rocm::SkipLayerNormTunableOp<__half, float, __half, true>": {
                        "4096_4620288": 20,
                        "4096_13697024": 39,
                        "4096_16744448": 39,
                    },
                },
                "validators": {
                    "ORT_VERSION": "1.17.0",
                    "ORT_GIT_COMMIT": "",
                    "ORT_BUILD_CONFIG": "USE_CK=1|USE_ROCBLAS_EXTENSION_API=1|USE_HIPBLASLT=1|",
                    "HIP_VERSION": "50731921",
                    "ROCBLAS_VERSION": "3.1.0.b80e4220-dirty",
                    "DEVICE_MODEL": "AMD Instinct MI250X/MI250",
                },
            }
        ]

        mock = MagicMock()
        mock.get_providers.return_value = ["ROCMExecutionProvider"]
        mock.get_tuning_results.return_value = tuning_result
        inference_session_mock.return_value = mock

        # setup
        model = get_onnx_model()
        model.inference_settings = {}
        model.inference_settings["tuning_op_result"] = tuning_result
        evaluator = OnnxEvaluator()
        latency_metric = get_latency_metric(LatencySubType.AVG)
        evaluator.evaluate(model, [latency_metric], Device.GPU, ["ROCMExecutionProvider"])
        mock.set_tuning_results.assert_called_with(tuning_result)

    @pytest.mark.parametrize(
        ("metric_inference_settings", "model_inference_settings", "result_keys"),
        [
            (None, None, None),
            (
                None,
                {
                    "execution_provider": ["ROCMExecutionProvider"],
                    "provider_options": [{"tunable_op_enable": True, "tunable_op_tuning_enable": True, "device_id": 0}],
                },
                ["execution_provider", "provider_options"],
            ),
            (
                {
                    "session_options": {
                        "intra_op_num_threads": 1,
                        "inter_op_num_threads": 0,
                        "execution_mode": 0,
                        "graph_opt_level": 99,
                    }
                },
                None,
                ["session_options"],
            ),
            (
                {"session_options": {"enable_profiling": True}},
                {"tuning_op_result": [{"ep": "ROCMExecutionProvider"}], "session_options": {"enable_profiling": False}},
                ["session_options", "tuning_op_result"],
            ),
        ],
    )
    def test_evaluator_get_inference_session(self, metric_inference_settings, model_inference_settings, result_keys):
        """Test get_inference_session method in evaluator when both metric and model have inference settings.

        The model.inference_settings will be overridden by the metric.inference_settings.
        """
        metric = get_latency_metric(LatencySubType.AVG)
        if metric_inference_settings:
            metric.user_config.inference_settings = {"onnx": metric_inference_settings.copy()}
        model = get_onnx_model()
        model.inference_settings = model_inference_settings.copy() if model_inference_settings else None
        inference_settings = OnnxEvaluator.get_inference_settings(metric, model)
        if result_keys is None:
            assert inference_settings == {}  # pylint: disable=use-implicit-booleaness-not-comparison
        else:
            for key in result_keys:
                assert key in inference_settings
                value = None
                if metric_inference_settings:
                    value = metric_inference_settings.get(key)
                if value is None and model_inference_settings:
                    value = model_inference_settings.get(key)
                assert inference_settings[key] == value
            if metric_inference_settings and model_inference_settings:
                # verify the metric inference settings has higher priority
                assert inference_settings["session_options"]["enable_profiling"]
                # verify the original inference settings are not changed
                assert metric.get_inference_settings("onnx") == metric_inference_settings
                assert model.inference_settings == model_inference_settings


class TestOliveEvaluatorConfig:
    @pytest.mark.parametrize(
        ("priorities", "has_exception"), [((1, 2), False), ((1, 1), True), ((2, 1), False), ((1, 0), True)]
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

    @pytest.mark.parametrize(
        ("metric_args", "is_accuracy_drop_tolerance"),
        [
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "min-improvement"}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "percent-min-improvement"}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "max-degradation"}}], True),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "percent-max-degradation"}}], True),
        ],
    )
    def test_is_accuracy_drop_tolerance(self, metric_args, is_accuracy_drop_tolerance):
        evaluator_config = [get_accuracy_metric(*m_arg["args"], **m_arg["kwargs"]) for m_arg in metric_args]
        evaluator_config_instance = OliveEvaluatorConfig(metrics=evaluator_config)
        assert evaluator_config_instance.is_accuracy_drop_tolerance == is_accuracy_drop_tolerance
