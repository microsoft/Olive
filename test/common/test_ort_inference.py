# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import sys
from unittest.mock import patch

import pytest

from olive.common.ort_inference import (
    get_ort_available_providers,
    maybe_register_ep_libraries,
    ort_supports_ep_devices,
)


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


def test_get_ort_available_providers_raises_actionable_error_when_onnxruntime_missing():
    with (
        patch.dict("sys.modules", {"onnxruntime": None}),
        pytest.raises(ImportError, match=r"onnxruntime is not installed.*olive-ai\[cpu\]"),
    ):
        get_ort_available_providers()


def test_get_ort_available_providers_keeps_original_error_when_dependency_missing():
    def fake_import(name, *args, **kwargs):
        if name == "onnxruntime":
            raise ModuleNotFoundError("No module named 'some_dependency'", name="some_dependency")
        return real_import(name, *args, **kwargs)

    real_import = __import__
    with patch.dict("sys.modules"), patch("builtins.__import__", side_effect=fake_import):
        sys.modules.pop("onnxruntime", None)
        with pytest.raises(ModuleNotFoundError, match="some_dependency"):
            get_ort_available_providers()
