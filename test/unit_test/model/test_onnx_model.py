from test.unit_test.utils import get_onnx_model
from unittest.mock import ANY, MagicMock, patch

import pytest

from olive.exception import OliveEvaluationError
from olive.hardware.accelerator import Device


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_model_prepare_session(mock_get_available_providers, inference_session_mock):
    mock_get_available_providers.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
    ]
    mock = MagicMock()
    mock.get_providers.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
        "CUDAExecutionProvider",
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

    execution_providers = ["ROCMExecutionProvider", "MIGraphXExecutionProvider"]
    provider_options = [{}, {}]
    # The inference session ep priority is lower than specified by argument in DLIS scenarios
    _ = model.prepare_session(inference_settings, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_once_with(
        model.model_path, sess_options=ANY, providers=execution_providers, provider_options=provider_options
    )


@patch("onnxruntime.InferenceSession")
@patch("onnxruntime.get_available_providers")
def test_model_prepare_session_with_unsupported_eps(mock_get_available_providers, inference_session_mock):
    mock_get_available_providers.return_value = [
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
    provider_options = [{}, {"device_id": "1"}]
    with pytest.raises(
        OliveEvaluationError,
        match=(
            "The onnxruntime fallback happens. TensorrtExecutionProvider is not in the session providers"
            r" \['CUDAExecutionProvider', 'CPUExecutionProvider'\]"
        ),
    ):
        # The inference session ep priority is lower than specified by argument in DLIS scenarios
        _ = model.prepare_session(inference_settings, Device.GPU, execution_providers, rank=1)
        inference_session_mock.assert_called_once_with(
            model.model_path, sess_options=ANY, providers=execution_providers, provider_options=provider_options
        )
