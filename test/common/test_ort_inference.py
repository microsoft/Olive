# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from types import SimpleNamespace
from unittest.mock import patch

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


@pytest.mark.skipif(not ort_supports_ep_devices(), reason="ORT does not support EP devices")
@pytest.mark.parametrize(
    ("ep_path_map", "expected_path"),
    [
        ({"OpenVINOExecutionProvider": None}, "path/to/openvino_plugin.dll"),
        ({"OpenVINOExecutionProvider": "path/to/explicit.dll"}, "path/to/explicit.dll"),
    ],
)
@patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
def test_maybe_register_ep_libraries_discovers_openvino_plugin(_, monkeypatch, ep_path_map, expected_path):
    monkeypatch.setitem(
        sys.modules, "onnxruntime_ep_openvino", SimpleNamespace(get_library_path=lambda: "path/to/openvino_plugin.dll")
    )
    with patch("onnxruntime.register_execution_provider_library") as mock_register:
        maybe_register_ep_libraries(ep_path_map)

    mock_register.assert_called_once_with("OpenVINOExecutionProvider", expected_path)


@pytest.mark.skipif(not ort_supports_ep_devices(), reason="ORT does not support EP devices")
@pytest.mark.parametrize("ep_path_map", [{"OpenVINOExecutionProvider": None}, {"CPUExecutionProvider": None}])
@patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider", "OpenVINOExecutionProvider"])
@patch("pathlib.Path.exists", new=lambda value: str(value).endswith("onnxruntime_providers_openvino.dll"))
def test_maybe_register_ep_libraries_ignores_builtin_openvino_when_deprecated_wheel_installed(
    _, monkeypatch, ep_path_map
):
    monkeypatch.setitem(
        sys.modules, "onnxruntime_ep_openvino", SimpleNamespace(get_library_path=lambda: "path/to/openvino_plugin.dll")
    )
    with patch("onnxruntime.register_execution_provider_library") as mock_register:
        maybe_register_ep_libraries(ep_path_map)

    registered = [call.args for call in mock_register.call_args_list]
    assert ("OpenVINOExecutionProvider", "onnxruntime_providers_openvino.dll") not in registered
    if "OpenVINOExecutionProvider" in ep_path_map:
        assert registered == [("OpenVINOExecutionProvider", "path/to/openvino_plugin.dll")]
    else:
        assert registered == []


@pytest.mark.skipif(not ort_supports_ep_devices(), reason="ORT does not support EP devices")
@pytest.mark.parametrize("builtin_library_exists", [True, False])
@patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider", "OpenVINOExecutionProvider"])
def test_maybe_register_ep_libraries_warns_only_when_deprecated_openvino_wheel_installed(
    _, monkeypatch, caplog, builtin_library_exists
):
    monkeypatch.setitem(
        sys.modules, "onnxruntime_ep_openvino", SimpleNamespace(get_library_path=lambda: "path/to/openvino_plugin.dll")
    )
    with (
        patch("pathlib.Path.exists", return_value=builtin_library_exists),
        patch("onnxruntime.register_execution_provider_library"),
        caplog.at_level("WARNING"),
    ):
        maybe_register_ep_libraries({"OpenVINOExecutionProvider": None})

    assert ("onnxruntime-openvino is deprecated" in caplog.text) == builtin_library_exists


@pytest.mark.skipif(not ort_supports_ep_devices(), reason="ORT does not support EP devices")
@patch("onnxruntime.get_available_providers", return_value=["CPUExecutionProvider"])
def test_maybe_register_ep_libraries_skips_openvino_when_plugin_missing(_, monkeypatch):
    monkeypatch.setitem(sys.modules, "onnxruntime_ep_openvino", None)
    with patch("onnxruntime.register_execution_provider_library") as mock_register:
        maybe_register_ep_libraries({"OpenVINOExecutionProvider": None})

    mock_register.assert_not_called()
