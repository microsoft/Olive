# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from test.unit_test.utils import get_onnx_model
from unittest.mock import PropertyMock, patch

import pytest

from olive.common.config_utils import validate_config
from olive.data.config import DataComponentConfig, DataConfig
from olive.evaluator.metric_result import flatten_metric_result
from olive.evaluator.olive_evaluator import OliveEvaluator, OnnxEvaluator
from olive.hardware.accelerator import DEFAULT_GPU_CUDA_ACCELERATOR
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.session_params_tuning import OrtSessionParamsTuning


def _get_tuning_data_config(input_shapes, input_names=None):
    data_config = DataConfig(
        name="test_data_config_for_tuning",
        type="DummyDataContainer",
        load_dataset_config=DataComponentConfig(
            params={
                "input_shapes": input_shapes,
                "input_names": input_names,
            }
        ),
    )
    return validate_config(data_config, DataConfig)


@pytest.mark.parametrize(
    "config",
    [
        {},
        {"data_config": _get_tuning_data_config([(1, 1)])},
        {"data_config": _get_tuning_data_config([(1, 1)], ["input"])},
    ],
)
def test_ort_session_params_tuning_pass(config, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtSessionParamsTuning, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)


@patch("olive.passes.onnx.session_params_tuning.OrtSessionParamsTuning._run_for_config")
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"data_config": _get_tuning_data_config([(1, 1)], ["input"])},
        {"providers_list": "CUDAExecutionProvider", "device": "gpu"},
    ],
)
def test_ort_session_params_tuning_with_customized_configs(mock_run, config):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtSessionParamsTuning, config, disable_search=True)

    # execute
    p.run(input_model, None)

    # assert
    if "providers_list" not in config:
        assert (
            mock_run.call_args.args[1].providers_list == "CPUExecutionProvider"
        ), "providers_list is not set correctly as ['CPUExecutionProvider'] by default when user does not specify it"
    if "device" not in config:
        assert (
            mock_run.call_args.args[1].device == "cpu"
        ), "device is not set correctly as cpu by default when user does not specify it"
    for k, v in config.items():
        assert getattr(mock_run.call_args.args[1], k) == v, f"{k} is not set correctly as {v}"


@pytest.mark.parametrize(
    ("execution_provider", "provider_options"),
    [
        ("TensorrtExecutionProvider", {"trt_fp16_enable": True}),
        (
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
            },
        ),
        ("CPUExecutionProvider", {}),
    ],
)
@patch.object(OnnxEvaluator, "evaluate")
@patch("onnxruntime.get_available_providers")
def test_session_params_tuning_with_provider_options(
    get_available_providers_mock, evaluate_mock, execution_provider, provider_options
):
    logger = logging.getLogger("olive")
    logger.propagate = True
    get_available_providers_mock.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    def mock_evaluate_method(model, metrics, device, execution_providers):
        metrics_res = {}
        latency = 0.5

        assert execution_providers is None
        ep = metrics[0].user_config.inference_settings["onnx"]["execution_provider"]
        if len(ep) == 1:
            if ep[0] == "CUDAExecutionProvider":
                latency = 0.4
            elif ep[0] == "TensorrtExecutionProvider":
                latency = 0.7
            else:
                latency = 0.5

        latency_metric = OliveEvaluator.compute_latency(metrics[0], [latency])
        metrics_res[metrics[0].name] = latency_metric
        return flatten_metric_result(metrics_res)

    evaluate_mock.side_effect = mock_evaluate_method
    config = {
        "providers_list": execution_provider,
        "provider_options_list": provider_options,
        "device": "gpu",
        "enable_cuda_graph": True,
    }
    input_model = get_onnx_model()
    p = create_pass_from_dict(
        OrtSessionParamsTuning, config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR
    )
    result = p.run(input_model, None)
    assert "execution_provider" in result.inference_settings
    acutal_eps = result.inference_settings["execution_provider"]
    assert "io_bind" in result.inference_settings
    assert acutal_eps == [execution_provider]
    if execution_provider == "CUDAExecutionProvider":
        assert result.inference_settings["provider_options"][0][
            "enable_cuda_graph"
        ], "enable_cuda_graph should be overridden to True"
        assert result.inference_settings["provider_options"][0]["arena_extend_strategy"] == "kNextPowerOfTwo"
    else:
        assert "enable_cuda_graph" not in result.inference_settings["provider_options"][0]


@patch("olive.model.ONNXModelHandler.io_config", new_callable=PropertyMock)
def test_ort_session_params_tuning_pass_with_dynamic_shapes(mock_get_io_config, tmp_path):
    mock_get_io_config.return_value = {
        "input_names": ["input"],
        "input_shapes": [["input_0", "input_1"]],
        "input_types": ["float32", "float32"],
        "output_names": ["output"],
        "output_shapes": [["input_0", 10]],
        "output_types": ["float32", "float32"],
    }

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtSessionParamsTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(TypeError) as e:
        # execute
        p.run(input_model, output_folder)
    assert "ones() received an invalid combination of arguments" in str(e.value)
