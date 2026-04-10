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
        # execute
        result = _resolve_home_dir()

        # assert
        assert isinstance(result, Path)

    def test_with_home_env_set(self):
        # execute
        with patch.dict(os.environ, {"HOME": "/tmp/test_home"}):
            result = _resolve_home_dir()

        # assert
        assert isinstance(result, Path)

    def test_without_home_env(self):
        # execute
        with patch.dict(os.environ, {}, clear=True):
            result = _resolve_home_dir()

        # assert
        assert isinstance(result, Path)


class TestGetTelemetryBaseDir:
    def test_returns_path(self):
        # setup
        get_telemetry_base_dir.cache_clear()

        # execute
        result = get_telemetry_base_dir()

        # assert
        assert isinstance(result, Path)

    def test_path_contains_onnxruntime(self):
        # setup
        get_telemetry_base_dir.cache_clear()

        # execute
        result = get_telemetry_base_dir()

        # assert
        assert ".onnxruntime" in str(result)


class TestFormatExceptionMessage:
    def test_basic_exception(self):
        # setup
        try:
            1 / 0  # noqa: B018
        except ZeroDivisionError as ex:
            exception = ex
            traceback = ex.__traceback__

        # execute
        result = _format_exception_message(exception, traceback)

        # assert
        assert "ZeroDivisionError" in result

    def test_exception_without_traceback(self):
        # setup
        ex = ValueError("test error")

        # execute
        result = _format_exception_message(ex)

        # assert
        assert "test error" in result


class TestEncodeCacheLine:
    def test_encode_basic_string(self):
        # setup
        expected = base64.b64encode(b"hello").decode("ascii")

        # execute
        result = _encode_cache_line("hello")

        # assert
        assert result == expected

    def test_encode_empty_string(self):
        # setup
        expected = base64.b64encode(b"").decode("ascii")

        # execute
        result = _encode_cache_line("")

        # assert
        assert result == expected

    def test_encode_unicode_string(self):
        # execute
        result = _encode_cache_line("hello \u4e16\u754c")
        decoded = base64.b64decode(result).decode("utf-8")

        # assert
        assert decoded == "hello \u4e16\u754c"


class TestDecodeCacheLine:
    def test_decode_basic_string(self):
        # setup
        encoded = base64.b64encode(b"hello").decode("ascii")

        # execute
        result = _decode_cache_line(encoded)

        # assert
        assert result == "hello"

    def test_decode_empty_string(self):
        # setup
        encoded = base64.b64encode(b"").decode("ascii")

        # execute
        result = _decode_cache_line(encoded)

        # assert
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
        # execute
        encoded = _encode_cache_line(text)
        decoded = _decode_cache_line(encoded)

        # assert
        assert decoded == text
