# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import contextlib
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from olive.evaluator import ort_genai_provider


def test_configured_provider_names():
    config = {
        "model": {
            "decoder": {"session_options": {"provider_options": []}},
            "vision": {
                "session_options": {"provider_options": [{"QNN": {"htp_performance_mode": "burst"}}]},
                "pipeline": {
                    "projector": {
                        "session_options": {"provider_options": [{"OpenVINO": {}}]},
                    }
                },
            },
        }
    }

    assert ort_genai_provider._configured_provider_names(config) == {"QNN", "OpenVINO"}


def test_register_configured_execution_provider_libraries():
    ready_state = object()
    result = SimpleNamespace(status=ready_state, diagnostic_text="", extended_error=SimpleNamespace(value=0))
    provider = SimpleNamespace(
        name="QNNExecutionProvider",
        library_path="qnn-provider.dll",
        ensure_ready_async=lambda: SimpleNamespace(get=lambda: result),
    )
    catalog = SimpleNamespace(find_all_providers=lambda: [provider])
    winml = SimpleNamespace(
        ExecutionProviderCatalog=SimpleNamespace(get_default=lambda: catalog),
        ExecutionProviderReadyResultState=SimpleNamespace(SUCCESS=ready_state),
    )
    bootstrap = SimpleNamespace(
        InitializeOptions=SimpleNamespace(ON_NO_MATCH_SHOW_UI=object()),
        initialize=lambda **kwargs: contextlib.nullcontext(),
    )
    og = MagicMock()
    config = {"session_options": {"provider_options": [{"QNN": {}}]}}

    ort_genai_provider._REGISTERED_PROVIDER_LIBRARIES.clear()
    with (
        patch("platform.system", return_value="Windows"),
        patch.object(
            ort_genai_provider,
            "_get_windows_ml_api",
            return_value=(winml, bootstrap.InitializeOptions, bootstrap.initialize),
        ),
    ):
        ort_genai_provider.register_configured_execution_provider_libraries(og, config)

    og.register_execution_provider_library.assert_called_once_with(
        "QNNExecutionProvider",
        "qnn-provider.dll",
    )
