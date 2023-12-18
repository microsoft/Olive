from test.unit_test.utils import get_onnx_model
from unittest.mock import ANY, MagicMock, patch

from olive.hardware.accelerator import Device
from olive.model.utils.onnx_utils import check_ort_fallback


@patch("onnxruntime.InferenceSession")
def test_model_prepare_session(inference_session_mock):
    mock = MagicMock()
    mock.get_providers.return_value = [
        "MIGraphXExecutionProvider",
        "ROCMExecutionProvider",
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

    execution_providers = ["ROCMExecutionProvider", "MIGraphXExecutionProvider", "CUDAExecutionProvider"]
    provider_options = [{}, {}, {"device_id": "1"}]
    # The inference session ep priority is lower than specified by argument in DLIS scenarios
    session = model.prepare_session(inference_settings, Device.GPU, execution_providers, rank=1)
    inference_session_mock.assert_called_once_with(
        model.model_path, sess_options=ANY, providers=execution_providers, provider_options=provider_options
    )
    check_ort_fallback(session, execution_providers)
