# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.util
from functools import partial
from types import FunctionType
from typing import ClassVar
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from olive.evaluator.metric import AccuracySubType, LatencySubType, MetricType, ThroughputSubType
from olive.evaluator.olive_evaluator import (
    OliveEvaluator,
    OliveEvaluatorConfig,
    OnnxEvaluator,
    OpenVINOEvaluator,
    PyTorchEvaluator,
    _is_vision_metric,
    _validate_vision_task_metric,
)
from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import Device
from test.utils import (
    get_accuracy_metric,
    get_custom_metric,
    get_custom_metric_no_eval,
    get_latency_metric,
    get_mock_openvino_model,
    get_onnx_model,
    get_pytorch_model,
    get_throughput_metric,
)


class TestOliveEvaluator:
    ACCURACY_TEST_CASE: ClassVar[list] = [
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

    LATENCY_TEST_CASE: ClassVar[list] = [
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
            match=r"The onnxruntime fallback happens\. OpenVINOExecutionProvider is not in the session providers",
        ):
            evaluator.evaluate(model, [latency_metric], Device.CPU, execution_providers)

    THROUGHPUT_TEST_CASE: ClassVar[list] = [
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

    CUSTOM_TEST_CASE: ClassVar[list] = [
        (PyTorchEvaluator(), get_pytorch_model, get_custom_metric, 0.382715310),
        (OnnxEvaluator(), get_onnx_model, get_custom_metric, 0.382715310),
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
                    "onnxruntime::rocm::tunable::blas::internal::GemmTunableOp<__half, ck::tensor_layout::gemm::RowMajor, ck::tensor_layout::gemm::RowMajor>": {
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
            # Initialize user_config if needed to set inference_settings
            if metric.user_config is None:
                from olive.common.config_utils import ConfigBase

                metric.user_config = ConfigBase()
            # Use object.__setattr__ to set dynamic attributes on ConfigBase
            object.__setattr__(metric.user_config, "inference_settings", {"onnx": metric_inference_settings.copy()})
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
        ("metric_args", "is_accuracy_drop_tolerant"),
        [
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "min-improvement"}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "percent-min-improvement"}}], False),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "max-degradation"}}], True),
            ([{"args": [AccuracySubType.ACCURACY_SCORE], "kwargs": {"goal_type": "percent-max-degradation"}}], True),
        ],
    )
    def test_is_accuracy_drop_tolerant(self, metric_args, is_accuracy_drop_tolerant):
        evaluator_config = [get_accuracy_metric(*m_arg["args"], **m_arg["kwargs"]) for m_arg in metric_args]
        evaluator_config_instance = OliveEvaluatorConfig(metrics=evaluator_config)
        assert evaluator_config_instance.is_accuracy_drop_tolerant == is_accuracy_drop_tolerant

    @patch("olive.common.import_lib.import_user_module")
    @patch("olive.evaluator.registry.Registry.get")
    def test_valid_custom_type_validation(self, registry_get_mock, import_user_module_mock):
        registry_get_mock.return_value = MagicMock()
        OliveEvaluatorConfig.from_json({"type": "test_evaluator"})
        registry_get_mock.assert_called_once_with("test_evaluator")

    @patch("olive.common.import_lib.import_user_module")
    @patch("olive.evaluator.registry.Registry.get")
    def test_invalid_custom_type_validation(self, registry_get_mock, import_user_module_mock):
        registry_get_mock.return_value = None

        with pytest.raises(ValidationError):
            OliveEvaluatorConfig.from_json({"type": "test_evaluator"})

        registry_get_mock.assert_called_once_with("test_evaluator")


@pytest.mark.skipif(
    importlib.util.find_spec("lm_eval") is None,
    reason="lm_eval not installed",
)
class TestLMEvaluatorModelClass:
    """Verify LMEvaluator dispatches to the lm-eval model backend matching model_class."""

    @pytest.mark.parametrize("model_class", ["ort", "ortgenai"])
    @patch("lm_eval.utils.setup_logging")
    @patch("lm_eval.tasks.TaskManager")
    @patch("lm_eval.simple_evaluate")
    @patch("lm_eval.api.registry.get_model")
    def test_lm_evaluator_dispatches_to_requested_backend(
        self, get_model_mock, simple_evaluate_mock, _task_manager_mock, _setup_logging_mock, model_class
    ):
        from olive.evaluator.olive_evaluator import LMEvaluator
        from olive.model.handler.onnx import ONNXModelHandler

        simple_evaluate_mock.return_value = {"results": {}}
        get_model_mock.return_value = MagicMock(return_value=MagicMock())

        evaluator = LMEvaluator(tasks=["arc_easy"], model_class=model_class, batch_size=1, max_length=128)

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = "/tmp/model.onnx"

        evaluator.evaluate(model, metrics=[], device=Device.CPU, execution_providers=["CPUExecutionProvider"])

        get_model_mock.assert_called_once_with(model_class)


@pytest.mark.skipif(
    importlib.util.find_spec("lm_eval") is None,
    reason="lm_eval not installed",
)
class TestLMEvalORTGenAIChatTemplate:
    def _bare_instance(self, pretrained: str):
        # pylint: disable=protected-access
        from olive.evaluator.lmeval_ort import LMEvalORTGenAIEvaluator

        instance = object.__new__(LMEvalORTGenAIEvaluator)
        instance._pretrained = pretrained
        instance._hf_tokenizer = None
        return instance

    @pytest.mark.parametrize(
        ("pretrained", "expected"),
        [
            ("/models/lfm2-350m", "__models__lfm2-350m"),
            ("relative/path/model", "relative__path__model"),
            ("C:\\models\\lfm2-350m", "C:__models__lfm2-350m"),
        ],
    )
    def test_tokenizer_name_normalizes_separators(self, pretrained, expected):
        assert self._bare_instance(pretrained).tokenizer_name == expected

    @patch("olive.evaluator.lmeval_ort.AutoTokenizer")
    def test_apply_chat_template_lazy_loads_hf_tokenizer(self, auto_tokenizer_mock):
        chat_history = [{"role": "user", "content": "hello"}]
        mock_tokenizer = MagicMock()
        mock_tokenizer.apply_chat_template.return_value = "rendered prompt"
        auto_tokenizer_mock.from_pretrained.return_value = mock_tokenizer

        instance = self._bare_instance("/models/lfm2")

        auto_tokenizer_mock.from_pretrained.assert_not_called()
        assert instance.apply_chat_template(chat_history) == "rendered prompt"
        auto_tokenizer_mock.from_pretrained.assert_called_once_with("/models/lfm2")

        instance.apply_chat_template(chat_history, add_generation_prompt=False)
        auto_tokenizer_mock.from_pretrained.assert_called_once()
        mock_tokenizer.apply_chat_template.assert_called_with(
            chat_history,
            tokenize=False,
            add_generation_prompt=False,
            continue_final_message=True,
        )


class TestVisionMetricValidation:
    """Tests for vision metric detection and task/metric validation."""

    def _make_vision_metric(self, sub_type_names, task=None):
        """Create a metric with vision sub-types and optional task param."""
        metric = MagicMock()
        metric.type = MetricType.ACCURACY
        metric.sub_types = [MagicMock(name=n) for n in sub_type_names]
        # MagicMock.name is special, set it explicitly
        for st, name in zip(metric.sub_types, sub_type_names):
            st.name = name

        if task is not None:
            metric.data_config.pre_process_data_config.type = "vision_vqa_pre_process"
            metric.data_config.pre_process_data_config.params = {"task": task}
        else:
            metric.data_config = None
        return metric

    def test_is_vision_metric_with_exact_match(self):
        metric = self._make_vision_metric(["exact_match"])
        assert _is_vision_metric(metric) is True

    def test_is_vision_metric_with_relaxed_accuracy(self):
        metric = self._make_vision_metric(["relaxed_accuracy"])
        assert _is_vision_metric(metric) is True

    def test_is_vision_metric_with_word_sort_ratio(self):
        metric = self._make_vision_metric(["word_sort_ratio"])
        assert _is_vision_metric(metric) is True

    def test_is_vision_metric_returns_false_for_standard(self):
        metric = self._make_vision_metric(["accuracy_score"])
        assert _is_vision_metric(metric) is False

    def test_is_vision_metric_raises_on_mixed_subtypes(self):
        with pytest.raises(ValueError, match="Cannot mix vision accuracy sub-types"):
            _is_vision_metric(self._make_vision_metric(["exact_match", "accuracy_score"]))

    def test_validate_vision_task_metric_compatible(self):
        metric = self._make_vision_metric(["exact_match"], task="vision-vqa")
        # Should not raise
        _validate_vision_task_metric(metric)

    def test_validate_vision_task_metric_incompatible(self):
        metric = self._make_vision_metric(["exact_match"], task="vision-ocr")
        with pytest.raises(ValueError, match="not compatible with task type"):
            _validate_vision_task_metric(metric)

    def test_validate_vision_task_metric_unknown_task(self):
        metric = self._make_vision_metric(["exact_match"], task="unknown-task")
        with pytest.raises(ValueError, match="Unknown vision task type"):
            _validate_vision_task_metric(metric)

    def test_validate_vision_task_metric_no_task_skips(self):
        metric = self._make_vision_metric(["exact_match"])
        # No task specified, should not raise
        _validate_vision_task_metric(metric)


class TestOnnxEvaluatorGenaiVisionDetection:
    """Tests for genai vision model detection and dispatch via the public evaluate() method."""

    def _make_model_with_genai_config(self, tmp_path, genai_config_content):
        """Create a mock ONNXModelHandler with a genai_config.json in its directory."""
        import json

        from olive.model.handler.onnx import ONNXModelHandler

        model_dir = tmp_path / "model"
        model_dir.mkdir()
        model_file = model_dir / "text.onnx"
        model_file.write_text("")  # dummy file

        if genai_config_content is not None:
            config_path = model_dir / "genai_config.json"
            config_path.write_text(json.dumps(genai_config_content))

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = str(model_file)
        model.framework = "onnx"
        return model

    def _make_vision_accuracy_metric(self):
        """Create a metric that triggers the vision accuracy evaluation path."""
        metric = MagicMock()
        metric.name = "accuracy"
        metric.type = MetricType.ACCURACY
        metric.sub_types = [MagicMock()]
        metric.sub_types[0].name = "exact_match"
        metric.data_config = None
        metric.user_config = MagicMock()
        metric.user_config.user_script = None
        metric.user_config.script_dir = None
        metric.user_config.data_dir = None
        metric.user_config.batch_size = 1
        metric.user_config.dataloader_func = None
        metric.user_config.post_processing_func = None
        metric.user_config.evaluate_func = None
        metric.user_config.input_names = None
        metric.user_config.input_shapes = None
        metric.backend = "huggingface_metrics"
        metric.sample_log_num = 0
        metric.sample_log_dir = None
        return metric

    def test_genai_vision_detected_when_vision_field_present(self, tmp_path):
        """Dispatch to genai vision path when genai_config.json has a vision field."""
        from olive.evaluator.olive_evaluator import OliveModelOutput

        config = {"model": {"vision": {"inputs": "pixel_values"}}}
        model = self._make_model_with_genai_config(tmp_path, config)

        with (
            patch.object(OnnxEvaluator, "_inference_vision_genai") as mock_genai,
            patch.object(OnnxEvaluator, "_inference_vision") as mock_vision,
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy") as mock_compute,
            patch("olive.evaluator.olive_evaluator._is_vision_metric", return_value=True),
            patch("olive.evaluator.olive_evaluator._validate_vision_task_metric"),
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.get_user_config") as mock_get_cfg,
            patch(
                "olive.evaluator.olive_evaluator.OliveEvaluator.generate_metric_user_config_with_model_io"
            ) as mock_gen,
        ):
            mock_genai.return_value = (OliveModelOutput(preds=["answer"], logits=None), ["answer"])
            mock_compute.return_value = MagicMock()
            metric = self._make_vision_accuracy_metric()
            mock_gen.return_value = metric
            mock_get_cfg.return_value = (MagicMock(), None, None)

            evaluator = OnnxEvaluator()
            evaluator.evaluate(model, [metric], Device.CPU, None)

            mock_genai.assert_called_once()
            mock_vision.assert_not_called()

    def test_genai_vision_detected_with_empty_vision_object(self, tmp_path):
        """Dispatch to genai vision path even when vision value is an empty dict."""
        from olive.evaluator.olive_evaluator import OliveModelOutput

        config = {"model": {"vision": {}}}
        model = self._make_model_with_genai_config(tmp_path, config)

        with (
            patch.object(OnnxEvaluator, "_inference_vision_genai") as mock_genai,
            patch.object(OnnxEvaluator, "_inference_vision") as mock_vision,
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy") as mock_compute,
            patch("olive.evaluator.olive_evaluator._is_vision_metric", return_value=True),
            patch("olive.evaluator.olive_evaluator._validate_vision_task_metric"),
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.get_user_config") as mock_get_cfg,
            patch(
                "olive.evaluator.olive_evaluator.OliveEvaluator.generate_metric_user_config_with_model_io"
            ) as mock_gen,
        ):
            mock_genai.return_value = (OliveModelOutput(preds=["answer"], logits=None), ["answer"])
            mock_compute.return_value = MagicMock()
            metric = self._make_vision_accuracy_metric()
            mock_gen.return_value = metric
            mock_get_cfg.return_value = (MagicMock(), None, None)

            evaluator = OnnxEvaluator()
            evaluator.evaluate(model, [metric], Device.CPU, None)

            mock_genai.assert_called_once()
            mock_vision.assert_not_called()

    def test_standard_vision_when_no_vision_field(self, tmp_path):
        """Dispatch to standard vision path when genai_config has no vision field."""
        from olive.evaluator.olive_evaluator import OliveModelOutput

        config = {"model": {"type": "whisper"}}
        model = self._make_model_with_genai_config(tmp_path, config)

        with (
            patch.object(OnnxEvaluator, "_inference_vision_genai") as mock_genai,
            patch.object(OnnxEvaluator, "_inference_vision") as mock_vision,
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy") as mock_compute,
            patch("olive.evaluator.olive_evaluator._is_vision_metric", return_value=True),
            patch("olive.evaluator.olive_evaluator._validate_vision_task_metric"),
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.get_user_config") as mock_get_cfg,
            patch(
                "olive.evaluator.olive_evaluator.OliveEvaluator.generate_metric_user_config_with_model_io"
            ) as mock_gen,
        ):
            mock_vision.return_value = (OliveModelOutput(preds=["answer"], logits=None), ["answer"])
            mock_compute.return_value = MagicMock()
            metric = self._make_vision_accuracy_metric()
            mock_gen.return_value = metric
            mock_get_cfg.return_value = (MagicMock(), None, None)

            evaluator = OnnxEvaluator()
            evaluator.evaluate(model, [metric], Device.CPU, None)

            mock_vision.assert_called_once()
            mock_genai.assert_not_called()

    def test_standard_vision_when_no_genai_config(self, tmp_path):
        """Dispatch to standard vision path when genai_config.json is missing."""
        from olive.evaluator.olive_evaluator import OliveModelOutput

        model = self._make_model_with_genai_config(tmp_path, None)

        with (
            patch.object(OnnxEvaluator, "_inference_vision_genai") as mock_genai,
            patch.object(OnnxEvaluator, "_inference_vision") as mock_vision,
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.compute_accuracy") as mock_compute,
            patch("olive.evaluator.olive_evaluator._is_vision_metric", return_value=True),
            patch("olive.evaluator.olive_evaluator._validate_vision_task_metric"),
            patch("olive.evaluator.olive_evaluator.OliveEvaluator.get_user_config") as mock_get_cfg,
            patch(
                "olive.evaluator.olive_evaluator.OliveEvaluator.generate_metric_user_config_with_model_io"
            ) as mock_gen,
        ):
            mock_vision.return_value = (OliveModelOutput(preds=["answer"], logits=None), ["answer"])
            mock_compute.return_value = MagicMock()
            metric = self._make_vision_accuracy_metric()
            mock_gen.return_value = metric
            mock_get_cfg.return_value = (MagicMock(), None, None)

            evaluator = OnnxEvaluator()
            evaluator.evaluate(model, [metric], Device.CPU, None)

            mock_vision.assert_called_once()
            mock_genai.assert_not_called()


class TestFindGenaiConfig:
    """Tests for _find_genai_config upward search behavior."""

    def test_find_genai_config_same_directory(self, tmp_path):
        """Find genai_config.json in the same directory as the ONNX file."""
        import json

        from olive.evaluator.olive_evaluator import _find_genai_config
        from olive.model.handler.onnx import ONNXModelHandler

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        onnx_file = model_dir / "model.onnx"
        onnx_file.write_text("")
        config_path = model_dir / "genai_config.json"
        config_path.write_text(json.dumps({"model": {"type": "test"}}))

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = str(onnx_file)

        result = _find_genai_config(model)
        assert result == config_path

    def test_find_genai_config_parent_directory(self, tmp_path):
        """Find genai_config.json one level up (nested model layout)."""
        import json

        from olive.evaluator.olive_evaluator import _find_genai_config
        from olive.model.handler.onnx import ONNXModelHandler

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        decoder_dir = model_dir / "decoder"
        decoder_dir.mkdir()
        onnx_file = decoder_dir / "model.onnx"
        onnx_file.write_text("")
        config_path = model_dir / "genai_config.json"
        config_path.write_text(json.dumps({"model": {"type": "gemma4"}}))

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = str(onnx_file)

        result = _find_genai_config(model)
        assert result == config_path

    def test_find_genai_config_not_found(self, tmp_path):
        """Return None when genai_config.json does not exist."""
        from olive.evaluator.olive_evaluator import _find_genai_config
        from olive.model.handler.onnx import ONNXModelHandler

        decoder_dir = tmp_path / "models" / "decoder"
        decoder_dir.mkdir(parents=True)
        onnx_file = decoder_dir / "model.onnx"
        onnx_file.write_text("")

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = str(onnx_file)

        result = _find_genai_config(model)
        assert result is None

    def test_find_genai_config_ignores_directory(self, tmp_path):
        """Ignore a directory named genai_config.json."""
        from olive.evaluator.olive_evaluator import _find_genai_config
        from olive.model.handler.onnx import ONNXModelHandler

        model_dir = tmp_path / "models"
        model_dir.mkdir()
        # Create a directory (not file) named genai_config.json
        fake_dir = model_dir / "genai_config.json"
        fake_dir.mkdir()

        decoder_dir = model_dir / "decoder"
        decoder_dir.mkdir()
        onnx_file = decoder_dir / "model.onnx"
        onnx_file.write_text("")

        model = MagicMock(spec=ONNXModelHandler)
        model.model_path = str(onnx_file)

        # Should not find the directory, should return None
        result = _find_genai_config(model)
        assert result is None


class TestSaveSampleLog:
    """Tests for OliveEvaluator.save_sample_log."""

    @staticmethod
    def _make_metric(sample_log_num=0, sample_log_dir=None, name="test_metric"):
        metric = MagicMock()
        metric.name = name
        metric.sample_log_num = sample_log_num
        metric.sample_log_dir = sample_log_dir
        return metric

    def test_save_sample_log_disabled_when_zero(self, tmp_path):
        """No file should be created when sample_log_num=0."""
        import torch

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=0, sample_log_dir=str(tmp_path), name="m")
        output = OliveModelOutput(preds=torch.tensor([1, 2, 3]), logits=None)
        targets = torch.tensor([1, 2, 3])

        OliveEvaluator.save_sample_log(metric, output, targets, 0)
        assert not list(tmp_path.iterdir())

    def test_save_sample_log_with_tensor_data(self, tmp_path):
        """Should write a JSONL file with tensor preds/targets converted to Python values."""
        import json

        import torch

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=3, sample_log_dir=str(tmp_path), name="accuracy")
        preds = torch.tensor([0, 1, 1, 0, 1])
        targets = torch.tensor([0, 1, 0, 0, 1])
        output = OliveModelOutput(preds=preds, logits=None)

        OliveEvaluator.save_sample_log(metric, output, targets, 3)

        log_path = tmp_path / "accuracy_samples.jsonl"
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 3

        for i, line in enumerate(lines):
            record = json.loads(line)
            assert record["index"] == i
            assert record["prediction"] == preds[i].item()
            assert record["target"] == targets[i].item()

    def test_save_sample_log_with_string_data(self, tmp_path):
        """Should handle string predictions and targets (text-based metrics)."""
        import json

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=2, sample_log_dir=str(tmp_path), name="wer")
        preds = ["hello world", "foo bar"]
        targets = ["hello world", "foo baz"]
        output = OliveModelOutput(preds=preds, logits=None)

        OliveEvaluator.save_sample_log(metric, output, targets, 2)

        log_path = tmp_path / "wer_samples.jsonl"
        assert log_path.exists()

        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        record0 = json.loads(lines[0])
        assert record0["prediction"] == "hello world"
        assert record0["target"] == "hello world"

        record1 = json.loads(lines[1])
        assert record1["prediction"] == "foo bar"
        assert record1["target"] == "foo baz"

    def test_save_sample_log_caps_at_available_samples(self, tmp_path):
        """When sample_log_num > len(preds), should write only available samples."""
        import torch

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=100, sample_log_dir=str(tmp_path), name="acc")
        preds = torch.tensor([1, 2])
        targets = torch.tensor([1, 0])
        output = OliveModelOutput(preds=preds, logits=None)

        OliveEvaluator.save_sample_log(metric, output, targets, 100)

        log_path = tmp_path / "acc_samples.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_save_sample_log_merges_extras(self, tmp_path):
        """Per-sample extras (e.g. prompt and image/audio file name) should be merged into records."""
        import json

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=2, sample_log_dir=str(tmp_path), name="vision_accuracy")
        preds = ["1", "3"]
        targets = ["3", "3"]
        extras = [
            {"prompt": "What is shown?\n1. cat\n2. dog", "image": "img_0.png"},
            {"prompt": "Which arrow?\n1. up\n2. down", "image": "img_1.png"},
        ]
        output = OliveModelOutput(preds=preds, logits=None, extras=extras)

        OliveEvaluator.save_sample_log(metric, output, targets, 2)

        log_path = tmp_path / "vision_accuracy_samples.jsonl"
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        record0 = json.loads(lines[0])
        # index first, then merged extras, then prediction/target
        assert list(record0.keys()) == ["index", "prompt", "image", "prediction", "target"]
        assert record0["prompt"] == "What is shown?\n1. cat\n2. dog"
        assert record0["image"] == "img_0.png"
        assert record0["prediction"] == "1"
        assert record0["target"] == "3"

        record1 = json.loads(lines[1])
        assert record1["image"] == "img_1.png"

    def test_save_sample_log_without_extras_is_unchanged(self, tmp_path):
        """When extras is None, records should only contain index/prediction/target."""
        import json

        from olive.evaluator.olive_evaluator import OliveModelOutput

        metric = self._make_metric(sample_log_num=1, sample_log_dir=str(tmp_path), name="acc")
        output = OliveModelOutput(preds=["a"], logits=None)

        OliveEvaluator.save_sample_log(metric, output, ["a"], 1)

        record = json.loads((tmp_path / "acc_samples.jsonl").read_text().strip())
        assert list(record.keys()) == ["index", "prediction", "target"]


class TestAudioInputHelpers:
    """Tests for the speech input normalization/unwrap helpers."""

    def test_normalize_audio_batch_dict_with_file_name(self):
        import numpy as np

        from olive.evaluator.olive_evaluator import _normalize_audio_batch

        arr = np.zeros(16000, dtype=np.float32)
        arrays, names = _normalize_audio_batch({"audio": np.expand_dims(arr, 0), "file_name": "a.wav"})
        assert len(arrays) == 1
        assert arrays[0].shape == (16000,)
        assert names == ["a.wav"]

    def test_normalize_audio_batch_legacy_array(self):
        import numpy as np

        from olive.evaluator.olive_evaluator import _normalize_audio_batch

        arr = np.zeros((1, 16000), dtype=np.float32)
        arrays, names = _normalize_audio_batch(arr)
        assert len(arrays) == 1
        assert names == [None]

    def test_unwrap_audio_input_dict(self):
        import numpy as np

        from olive.evaluator.olive_evaluator import _unwrap_audio_input

        arr = np.zeros((1, 8), dtype=np.float32)
        unwrapped = _unwrap_audio_input({"audio": arr, "file_name": "a.wav"})
        assert unwrapped is arr

    def test_unwrap_audio_input_passthrough(self):
        import numpy as np

        from olive.evaluator.olive_evaluator import _unwrap_audio_input

        arr = np.zeros((1, 8), dtype=np.float32)
        assert _unwrap_audio_input(arr) is arr
