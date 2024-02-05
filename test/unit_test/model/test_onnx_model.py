from test.unit_test.utils import get_onnx_model
from unittest.mock import ANY, MagicMock, patch

import pytest

from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import Device


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_model_prepare_session(get_available_providers_mock, inference_session_mock):
    get_available_providers_mock.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]
    mock = MagicMock()
    mock.get_providers.return_value = ["MIGraphXExecutionProvider"]
    inference_session_mock.return_value = mock
    model = get_onnx_model()
    inference_settings = {
        # the EP in inference_settings has higher priority than the EP in arguments
        "execution_provider": [("MIGraphXExecutionProvider", {})],
        "session_options": {
            "execution_mode": 0,
            "graph_optimization_level": 99,
            "extra_session_config": None,
            "inter_op_num_threads": None,
            "intra_op_num_threads": 1,
        },
    }

    execution_providers = ["ROCMExecutionProvider", "MIGraphXExecutionProvider"]
    # The inference session ep priority is lower than specified by argument in DLIS scenarios
    _ = model.prepare_session(inference_settings, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_once_with(
        model.model_path, sess_options=ANY, providers=["MIGraphXExecutionProvider"], provider_options=[{}]
    )


@pytest.mark.parametrize(
    ("inference_setting", "model_inference_settings", "execution_providers", "merged_inference_settings"),
    [
        # Non inference_settings cases
        (
            None,
            None,
            None,
            {
                "execution_provider": ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [{}, {}, {}],
            },
        ),
        (
            None,
            None,
            "ROCMExecutionProvider",
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{}],
            },
        ),
        (
            None,
            None,
            ["ROCMExecutionProvider", "CPUExecutionProvider"],
            {
                "execution_provider": ["ROCMExecutionProvider", "CPUExecutionProvider"],
                "provider_options": [{}, {}],
            },
        ),
        (
            None,
            None,
            ("ROCMExecutionProvider", {"device_id": 0, "tunable_op_enable": True, "tunable_op_tuning_enable": True}),
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [
                    {"device_id": "0", "tunable_op_enable": "True", "tunable_op_tuning_enable": "True"}
                ],
            },
        ),
        # parameter inference_settings cases
        (
            {
                "execution_provider": [("ROCMExecutionProvider", {"device_id": 0})],
            },
            None,
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "0"}],
            },
        ),
        (
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": 1}],
            },
            None,
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "1"}],
            },
        ),
        # model inference_settings cases
        (
            None,
            {
                "execution_provider": [("ROCMExecutionProvider", {"device_id": 2})],
            },
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "2"}],
            },
        ),
        (
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": 3}],
            },
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "3"}],
            },
        ),
        # parameter inference_settings and model inference_settings cases
        (
            {
                "execution_provider": [("ROCMExecutionProvider", {"device_id": 4})],
            },
            {
                "execution_provider": ["MIGraphXExecutionProvider"],
            },
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "4"}],
            },
        ),
        (
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": 5}],
            },
            {
                "execution_provider": [("MIGraphXExecutionProvider", {})],
            },
            None,
            {
                "execution_provider": ["ROCMExecutionProvider"],
                "provider_options": [{"device_id": "5"}],
            },
        ),
    ],
)
@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_model_prepare_session_multiple_inference_settings(
    get_available_providers_mock,
    inference_session_mock,
    inference_setting,
    model_inference_settings,
    execution_providers,
    merged_inference_settings,
):
    get_available_providers_mock.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CPUExecutionProvider",
    ]
    mock = MagicMock()
    mock.get_providers.return_value = ["MIGraphXExecutionProvider", "ROCMExecutionProvider", "CPUExecutionProvider"]
    inference_session_mock.return_value = mock

    model_inference_settings_copy = model_inference_settings.copy() if model_inference_settings else None
    inference_settings_copy = inference_setting.copy() if inference_setting else None
    model = get_onnx_model()
    model.inference_settings = model_inference_settings
    model.prepare_session(inference_setting, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_with(
        model.model_path,
        sess_options=ANY,
        providers=merged_inference_settings["execution_provider"],
        provider_options=merged_inference_settings["provider_options"],
    )
    # assert the inference_settings and model.inference_settings are not changed when calling prepare_session
    assert model.inference_settings == model_inference_settings_copy
    assert inference_setting == inference_settings_copy


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_model_prepare_session_with_unsupported_eps(get_available_providers_mock, inference_session_mock):
    get_available_providers_mock.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    mock = MagicMock()
    mock.get_providers.return_value = [
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    inference_session_mock.return_value = mock
    model = get_onnx_model()
    inference_settings = {
        "execution_provider": [("MIGraphXExecutionProvider", {})],
        "session_options": {
            "execution_mode": 0,
            "graph_optimization_level": 99,
            "extra_session_config": None,
            "inter_op_num_threads": None,
            "intra_op_num_threads": 1,
        },
    }
    execution_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    # provider_options = [{}, {"device_id": "1"}]
    with pytest.raises(
        OliveEvaluationError,
        match=(
            "The onnxruntime fallback happens. MIGraphXExecutionProvider is not in the session providers"
            r" \['CUDAExecutionProvider', 'CPUExecutionProvider'\]"
        ),
    ):
        # The inference session ep priority is lower than specified by argument in DLIS scenarios
        _ = model.prepare_session(inference_settings, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_once_with(
        model.model_path, sess_options=ANY, providers=["MIGraphXExecutionProvider"], provider_options=[{}]
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_distributed_rank_prepare_session(get_available_providers_mock, inference_session_mock):
    get_available_providers_mock.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]

    mock = MagicMock()
    mock.get_providers.return_value = [
        "TensorrtExecutionProvider",
        "CUDAExecutionProvider",
        "CPUExecutionProvider",
    ]
    inference_session_mock.return_value = mock
    model = get_onnx_model()

    execution_providers = ["TensorrtExecutionProvider", "CUDAExecutionProvider"]
    provider_options = [{}, {"device_id": "1"}]
    # The inference session ep priority is lower than specified by argument in DLIS scenarios
    _ = model.prepare_session(None, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_once_with(
        model.model_path, sess_options=ANY, providers=execution_providers, provider_options=provider_options
    )
