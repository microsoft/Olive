# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import math
import re
from test.unit_test.utils import create_dataloader, get_onnx_model
from unittest.mock import MagicMock, PropertyMock, patch

import psutil
import pytest

from olive.evaluator.metric import flatten_metric_result
from olive.evaluator.olive_evaluator import OliveEvaluator, OnnxEvaluator
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR, AcceleratorSpec, Device
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.perf_tuning import PERFTUNING_BASELINE, OrtPerfTuning, PerfTuningRunner, generate_test_name


@pytest.mark.parametrize(
    "config",
    [
        {"input_names": ["input"], "input_shapes": [[1, 1]]},
        {},
        {"dataloader_func": create_dataloader},
        {"dataloader_func": create_dataloader, "dataloader_func_kwargs": {"dummy_kwarg": 1}},
    ],
)
def test_ort_perf_tuning_pass(config, tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


@patch("olive.passes.onnx.perf_tuning.OrtPerfTuning._run_for_config")
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"input_names": ["input"], "input_shapes": [[1, 1]]},
        {"providers_list": ["CPUExecutionProvider", "CUDAExecutionProvider"], "device": "gpu"},
    ],
)
def test_ort_perf_tuning_with_customized_configs(mock_run, config):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True)

    # execute
    p.run(input_model, None, None)

    # assert
    if "providers_list" not in config:
        assert mock_run.call_args.args[2]["providers_list"] == [
            "CPUExecutionProvider"
        ], "providers_list is not set correctly as ['CPUExecutionProvider'] by default when user does not specify it"
    if "device" not in config:
        assert (
            mock_run.call_args.args[2]["device"] == "cpu"
        ), "device is not set correctly as cpu by default when user does not specify it"
    for k, v in config.items():
        assert mock_run.call_args.args[2][k] == v, f"{k} is not set correctly as {v}"


@pytest.mark.parametrize("return_baseline", [True, False])
@patch.object(OnnxEvaluator, "evaluate")
@patch("onnxruntime.get_available_providers")
def test_perf_tuning_with_provider_options(get_available_providers_mock, evaluate_mock, caplog, return_baseline):
    logger = logging.getLogger("olive")
    logger.propagate = True
    get_available_providers_mock.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    def mock_evaluate_method(model, data_root, metrics, device, execution_providers):
        metrics_res = {}
        latency = 0.5

        assert execution_providers is None
        ep = metrics[0].user_config.inference_settings["onnx"]["execution_provider"]
        if len(ep) == 1:
            # for single perf tuning case
            if return_baseline:
                latency = 0.6
            else:
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
    execution_providers = [
        (
            "TensorrtExecutionProvider",
            {
                "trt_fp16_enable": True,
            },
        ),
        [
            "CUDAExecutionProvider",
            {
                "device_id": 0,
                "arena_extend_strategy": "kNextPowerOfTwo",
                "gpu_mem_limit": 2 * 1024 * 1024 * 1024,
                "cudnn_conv_algo_search": "EXHAUSTIVE",
                "do_copy_in_default_stream": True,
                "enable_cuda_graph": False,
            },
        ],
        "CPUExecutionProvider",
    ]
    config = {
        "providers_list": execution_providers,
        "device": "gpu",
        "enable_cuda_graph": True,
    }
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR)
    result = p.run(input_model, None, None)
    assert "execution_provider" in result.inference_settings
    acutal_eps = result.inference_settings["execution_provider"]
    assert "io_bind" in result.inference_settings
    if return_baseline:
        assert acutal_eps == ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        assert "enable_cuda_graph" not in result.inference_settings["provider_options"][0]
        assert re.search(f"Best result:.*{PERFTUNING_BASELINE}", caplog.text)
    else:
        assert len(acutal_eps) == 1
        assert acutal_eps[0] == "CUDAExecutionProvider"
        assert result.inference_settings["provider_options"][0][
            "enable_cuda_graph"
        ], "enable_cuda_graph is should be overridden to True"
        assert result.inference_settings["provider_options"][0]["arena_extend_strategy"] == "kNextPowerOfTwo"


@pytest.mark.parametrize("force_evaluate", [True, False])
@patch.object(OnnxEvaluator, "evaluate")
@patch("onnxruntime.get_available_providers")
def test_perf_tuning_with_force_evaluate(get_available_providers_mock, evaluate_mock, caplog, force_evaluate):
    logger = logging.getLogger("olive")
    logger.propagate = True

    get_available_providers_mock.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    def mock_evaluate_method(model, data_root, metrics, device, execution_providers):
        metrics_res = {}
        latency = 0.5
        assert execution_providers is None
        ep = metrics[0].user_config.inference_settings["onnx"]["execution_provider"]
        if len(ep) == 1:
            if ep[0] == "CPUExecutionProvider":
                latency = 0.5
            elif ep[0] == "CUDAExecutionProvider":
                latency = 0.1
        else:
            assert len(ep) == 2
            latency = 0.6

        latency_metric = OliveEvaluator.compute_latency(metrics[0], [latency])
        metrics_res[metrics[0].name] = latency_metric
        return flatten_metric_result(metrics_res)

    evaluate_mock.side_effect = mock_evaluate_method
    execution_providers = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    config = {
        "providers_list": execution_providers,
        "device": "gpu",
        "force_evaluate_other_eps": force_evaluate,
    }
    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, config, disable_search=True, accelerator_spec=DEFAULT_CPU_ACCELERATOR)
    result = p.run(input_model, None, None)
    assert "io_bind" in result.inference_settings
    if force_evaluate:
        assert "execution_provider" in result.inference_settings
        acutal_eps = result.inference_settings["execution_provider"]
        assert len(acutal_eps) == 1
        assert acutal_eps[0] == "CUDAExecutionProvider"
        assert re.search("Best result:.*cuda", caplog.text)
    else:
        assert "execution_provider" in result.inference_settings
        acutal_eps = result.inference_settings["execution_provider"]
        assert len(acutal_eps) == 1
        assert acutal_eps[0] == "CPUExecutionProvider"
        assert re.search("Best result: .*cpu", caplog.text)
        assert (
            "Ignore perf tuning for EP CUDAExecutionProvider since current pass EP is CPUExecutionProvider"
            in caplog.text
        )


@patch("olive.model.ONNXModelHandler.io_config", new_callable=PropertyMock)
def test_ort_perf_tuning_pass_with_dynamic_shapes(mock_get_io_config, tmp_path):
    mock_get_io_config.return_value = {
        "input_names": ["input"],
        "input_shapes": [["input_0", "input_1"]],
        "input_types": ["float32", "float32"],
        "output_names": ["output"],
        "output_shapes": [["input_0", 10]],
        "output_types": ["float32", "float32"],
    }

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(TypeError) as e:
        # execute
        p.run(input_model, None, output_folder)
    assert "ones() received an invalid combination of arguments" in str(e.value)


@patch.object(PerfTuningRunner, "threads_num_binary_search")
def test_ort_perf_tuning_pass_with_import_error(threads_num_binary_search_mock, tmp_path):
    threads_num_binary_search_mock.side_effect = ModuleNotFoundError("test")

    input_model = get_onnx_model()
    p = create_pass_from_dict(OrtPerfTuning, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    with pytest.raises(ModuleNotFoundError) as e:
        # execute
        p.run(input_model, None, output_folder)

    assert "test" in str(e.value)


def test_generate_test_name():
    test_params = {
        "execution_provider": ["CPUExecutionProvider"],
        "provider_options": [{}],
        "session_options": {
            "execution_mode": 1,
            "extra_session_config": None,
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 8,
        },
    }

    name = generate_test_name(test_params, True)
    assert name == (
        "cpu-{'execution_mode': 1, "
        "'extra_session_config': None, 'inter_op_num_threads': 1, 'intra_op_num_threads': 8}"
        "-{'io_bind': True}"
    )

    test_params = {
        "execution_provider": ["TensorrtExecutionProvider"],
        "provider_options": [{"trt_fp16_enable": True}],
        "session_options": {
            "execution_mode": 1,
            "extra_session_config": {
                "session.intra_op_thread_affinities": "0;1;2;3;4;5;6;7",
            },
            "inter_op_num_threads": 1,
            "intra_op_num_threads": 8,
            "graph_optimization_level": 99,
        },
    }
    name = generate_test_name(test_params, False)
    assert name == (
        "('tensorrt', {'trt_fp16_enable': True})-"
        "{'execution_mode': 1, "
        "'extra_session_config': {'session.intra_op_thread_affinities': '0;1;2;3;4;5;6;7'}, "
        "'inter_op_num_threads': 1, 'intra_op_num_threads': 8, 'graph_optimization_level': 99}"
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_rocm_tuning_enable(get_available_providers_mock, inference_session_mock, tmp_path):
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
    mock.get_providers.return_value = ["MIGraphXExecutionProvider", "ROCMExecutionProvider"]
    mock.get_tuning_results.return_value = tuning_result
    inference_session_mock.return_value = mock

    get_available_providers_mock.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]

    config = {
        "providers_list": ["MIGraphXExecutionProvider", "ROCMExecutionProvider"],
        "device": "gpu",
    }

    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(
        OrtPerfTuning,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec(accelerator_type=Device.GPU, execution_provider="ROCMExecutionProvider"),
    )
    output_folder = str(tmp_path / "onnx")

    # execute
    result = p.run(input_model, None, output_folder)
    tuning_result_ret = result.inference_settings["tuning_op_result"]
    assert tuning_result_ret == tuning_result
    set_tuning_result_binary_search_count_per_iteration = int(math.log2(psutil.cpu_count(logical=False))) + 1
    set_tuning_result_count = 3 * set_tuning_result_binary_search_count_per_iteration
    assert mock.set_tuning_results.call_count >= set_tuning_result_count
    assert mock.get_tuning_results.call_count == mock.set_tuning_results.call_count + 1
