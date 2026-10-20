# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from olive.common.ort_inference import maybe_register_ep_libraries, ort_supports_ep_devices


@pytest.mark.skipif(not ort_supports_ep_devices(), reason="ORT does not support EP devices")
@pytest.mark.parametrize(
    ("ep_path_map", "expected_call_list"),
    [
        ({"CPUExecutionProvider": None}, [("QNNExecutionProvider", "onnxruntime_providers_qnn.dll")]),
        ({"CUDAExecutionProvider": None}, [("QNNExecutionProvider", "onnxruntime_providers_qnn.dll")]),
        (
            {"CUDAExecutionProvider": "path/to/cuda/lib.dll"},
            [
                ("CUDAExecutionProvider", "path/to/cuda/lib.dll"),
                ("QNNExecutionProvider", "onnxruntime_providers_qnn.dll"),
            ],
        ),
        (
            {"CPUExecutionProvider": None, "CUDAExecutionProvider": "path/to/cuda/lib.dll"},
            [
                ("CUDAExecutionProvider", "path/to/cuda/lib.dll"),
                ("QNNExecutionProvider", "onnxruntime_providers_qnn.dll"),
            ],
        ),
        ({"QNNExecutionProvider": None}, [("QNNExecutionProvider", "onnxruntime_providers_qnn.dll")]),
        ({"QNNExecutionProvider": "path/to/qnn/lib.dll"}, [("QNNExecutionProvider", "path/to/qnn/lib.dll")]),
        (
            {"QNNExecutionProvider": None, "CUDAExecutionProvider": "path/to/cuda/lib.dll"},
            [
                ("CUDAExecutionProvider", "path/to/cuda/lib.dll"),
                ("QNNExecutionProvider", "onnxruntime_providers_qnn.dll"),
            ],
        ),
        (
            {"QNNExecutionProvider": "path/to/qnn/lib.dll", "CUDAExecutionProvider": "path/to/cuda/lib.dll"},
            [("CUDAExecutionProvider", "path/to/cuda/lib.dll"), ("QNNExecutionProvider", "path/to/qnn/lib.dll")],
        ),
    ],
)
@patch("onnxruntime.get_available_providers")
@patch("pathlib.Path.exists", new=lambda value: str(value).endswith("onnxruntime_providers_qnn.dll"))
def test_maybe_register_ep_libraries(get_available_providers_mock, ep_path_map, expected_call_list):
    get_available_providers_mock.return_value = ["CPUExecutionProvider", "QNNExecutionProvider"]
    with patch("onnxruntime.register_execution_provider_library") as mock_register:
        maybe_register_ep_libraries(ep_path_map)
        assert mock_register.call_count == len(expected_call_list)
        for ep_name, lib_path in expected_call_list:
            mock_register.assert_any_call(ep_name, lib_path)


@pytest.mark.parametrize("library_path", [None, "custom_openvino.dll"])
def test_maybe_register_ep_libraries_discovers_openvino_without_overriding_explicit_path(monkeypatch, library_path):
    get_library_path = MagicMock(return_value="plugin_openvino.dll")
    registration = MagicMock()
    monkeypatch.setattr("olive.common.ort_inference.ort_supports_ep_devices", lambda: True)
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_openvino", SimpleNamespace(get_library_path=get_library_path))
    monkeypatch.setattr("onnxruntime.get_available_providers", lambda: ["OpenVINOExecutionProvider"])
    monkeypatch.setattr("pathlib.Path.exists", lambda path: str(path).endswith("onnxruntime_providers_openvino.dll"))
    monkeypatch.setattr("onnxruntime.register_execution_provider_library", registration)

    maybe_register_ep_libraries({"OpenVINOExecutionProvider": library_path})

    registration.assert_called_once_with("OpenVINOExecutionProvider", library_path or "plugin_openvino.dll")
    if library_path:
        get_library_path.assert_not_called()
    else:
        get_library_path.assert_called_once_with()
