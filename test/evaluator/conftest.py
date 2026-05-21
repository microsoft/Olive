# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Inject a minimal onnxruntime_genai stub for generate_until unit tests.

Ensures tests can run in environments where the real package is not installed.
The tests mock all ORT GenAI objects anyway, so the stub only needs to provide
importable names.
"""

import importlib.util
import sys
import types
from unittest.mock import MagicMock


def _ensure_ort_genai_stub():
    if importlib.util.find_spec("onnxruntime_genai") is None:
        stub = types.ModuleType("onnxruntime_genai")
        stub.Generator = MagicMock
        stub.GeneratorParams = MagicMock
        sys.modules["onnxruntime_genai"] = stub


_ensure_ort_genai_stub()
