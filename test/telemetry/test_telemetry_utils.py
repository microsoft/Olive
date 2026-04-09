# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import base64
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.telemetry.utils import (
    _decode_cache_line,
    _encode_cache_line,
    _format_exception_message,
    _resolve_home_dir,
    get_telemetry_base_dir,
)


class TestResolveHomeDir:
    def test_returns_path(self):
        result = _resolve_home_dir()
        assert isinstance(result, Path)

    def test_with_home_env_set(self):
        with patch.dict(os.environ, {"HOME": "/tmp/test_home"}):
            result = _resolve_home_dir()
            assert isinstance(result, Path)

    def test_without_home_env(self):
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_home_dir()
            assert isinstance(result, Path)


class TestGetTelemetryBaseDir:
    def test_returns_path(self):
        # Clear the lru_cache before test
        get_telemetry_base_dir.cache_clear()
        result = get_telemetry_base_dir()
        assert isinstance(result, Path)

    def test_path_contains_onnxruntime(self):
        get_telemetry_base_dir.cache_clear()
        result = get_telemetry_base_dir()
        assert ".onnxruntime" in str(result)


class TestFormatExceptionMessage:
    def test_basic_exception(self):
        try:
            1 / 0  # noqa: B018
        except ZeroDivisionError as ex:
            result = _format_exception_message(ex, ex.__traceback__)
            assert "ZeroDivisionError" in result

    def test_exception_without_traceback(self):
        ex = ValueError("test error")
        result = _format_exception_message(ex)
        assert "test error" in result


class TestEncodeCacheLine:
    def test_encode_basic_string(self):
        result = _encode_cache_line("hello")
        expected = base64.b64encode(b"hello").decode("ascii")
        assert result == expected

    def test_encode_empty_string(self):
        result = _encode_cache_line("")
        expected = base64.b64encode(b"").decode("ascii")
        assert result == expected

    def test_encode_unicode_string(self):
        result = _encode_cache_line("hello 世界")
        decoded = base64.b64decode(result).decode("utf-8")
        assert decoded == "hello 世界"


class TestDecodeCacheLine:
    def test_decode_basic_string(self):
        encoded = base64.b64encode(b"hello").decode("ascii")
        result = _decode_cache_line(encoded)
        assert result == "hello"

    def test_decode_empty_string(self):
        encoded = base64.b64encode(b"").decode("ascii")
        result = _decode_cache_line(encoded)
        assert result == ""


class TestEncodeDecodeRoundtrip:
    @pytest.mark.parametrize(
        "text",
        [
            "simple text",
            "path/to/file.json",
            '{"key": "value"}',
            "special chars: !@#$%^&*()",
            "",
        ],
    )
    def test_roundtrip(self, text):
        encoded = _encode_cache_line(text)
        decoded = _decode_cache_line(encoded)
        assert decoded == text
