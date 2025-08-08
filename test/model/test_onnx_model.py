from unittest.mock import ANY, MagicMock, patch

import numpy as np
import onnx
import pytest

from olive.common.ort_inference import ort_supports_ep_devices
from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import Device
from olive.model import ONNXModelHandler
from test.utils import get_onnx_model


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
    if ort_supports_ep_devices():
        inference_session_mock.assert_called_once_with(model.model_path, sess_options=ANY)
    else:
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
@pytest.mark.skipif(ort_supports_ep_devices(), reason="onnxruntime.get_available_providers not supported")
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
@pytest.mark.skipif(ort_supports_ep_devices(), reason="onnxruntime.get_available_providers not supported")
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
@pytest.mark.skipif(ort_supports_ep_devices(), reason="onnxruntime.get_available_providers not supported")
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


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
@patch("onnxruntime.SessionOptions")
@patch("onnxruntime.OrtValue")
@pytest.mark.skipif(ort_supports_ep_devices(), reason="onnxruntime.get_available_providers not supported")
def test_model_prepare_session_with_external_initializers(
    ort_value_mock, session_options_mock, get_available_providers_mock, inference_session_mock, tmp_path
):
    # eps
    eps = ["CPUExecutionProvider"]

    # model
    model_proto = get_onnx_model().load_model()
    external_initializers = {}
    for initializer in model_proto.graph.initializer:
        external_initializers[initializer.name] = onnx.numpy_helper.to_array(initializer)

    # save model with external data
    onnx_file_name = "model.onnx"
    onnx.save(model_proto, tmp_path / onnx_file_name, save_as_external_data=True, location="model.onnx.data")

    # external initializers
    external_initializers_file_name = "external_initializers.npz"
    np.savez(tmp_path / external_initializers_file_name, **external_initializers)

    # create model
    olive_model = ONNXModelHandler(
        tmp_path, onnx_file_name=onnx_file_name, external_initializers_file_name=external_initializers_file_name
    )

    # mock
    get_available_providers_mock.return_value = eps

    # mock session options object so that we can check its method calls
    session_options = MagicMock()
    session_options_mock.return_value = session_options

    # mock ort value object so that we can check it was passed to add_external_initializers
    ort_value = MagicMock()
    ort_value_mock.ortvalue_from_numpy.return_value = ort_value

    # mock inference session object so that we can override the providers
    session = MagicMock()
    session.get_providers.return_value = eps
    inference_session_mock.return_value = session

    # test
    _ = olive_model.prepare_session(None, Device.CPU, eps, rank=1)

    # assert
    # check that the external initializers were added to the session options
    session_options.add_external_initializers.assert_called_once_with(
        list(external_initializers.keys()), [ort_value] * len(external_initializers)
    )
    # check that the correct external initializers were passed to the ortvalue_from_numpy method
    for actual_weight, expected_weight in zip(
        ort_value_mock.ortvalue_from_numpy.call_args_list, external_initializers.values()
    ):
        np.testing.assert_array_equal(actual_weight[0][0], expected_weight)
    # check that the session was created with the correct parameters
    inference_session_mock.assert_called_once_with(
        olive_model.model_path, sess_options=session_options, providers=eps, provider_options=[{}]
    )
