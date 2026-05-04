# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from olive.telemetry.library.telemetry_logger import TelemetryLogger
from olive.telemetry.telemetry import (
    ACTION_EVENT_NAME,
    CACHE_FILE_NAME,
    RECIPE_EVENT_NAME,
    Telemetry,
    TelemetryCacheHandler,
)
from olive.telemetry.utils import _exclusive_file_lock


def test_cache_path_uses_env_override(tmp_path, monkeypatch):
    cache_dir = tmp_path / "telemetry-cache"
    monkeypatch.setenv("OLIVE_TELEMETRY_CACHE_DIR", str(cache_dir))

    handler = TelemetryCacheHandler(Mock())

    assert handler.cache_path == cache_dir / CACHE_FILE_NAME
    assert isinstance(handler.cache_path, Path)


def test_telemetry_logger_uses_explicit_service_name():
    TelemetryLogger.shutdown_default_logger()
    TelemetryLogger._instance = None
    TelemetryLogger._default_logger = None

    try:
        logger = TelemetryLogger.get_default_logger(
            connection_string="InstrumentationKey=12345678-1234-1234-1234-123456789abc-tenant",
            service_name="Olive",
        )
        assert logger._logger_provider.resource.attributes["service.name"] == "Olive"
    finally:
        TelemetryLogger.shutdown_default_logger()
        TelemetryLogger._instance = None
        TelemetryLogger._default_logger = None


def test_telemetry_only_logs_recipe_events_in_ci(monkeypatch):
    monkeypatch.setenv("CI", "1")
    Telemetry._instance = None

    mock_logger = Mock()
    mock_logger.register_payload_transmitted_callback.return_value = lambda: None

    try:
        with patch("olive.telemetry.telemetry.get_telemetry_logger", return_value=mock_logger):
            telemetry = Telemetry()
            telemetry.log(ACTION_EVENT_NAME, {"action_name": "WorkflowRun", "duration_ms": 1, "success": False})
            telemetry.log(RECIPE_EVENT_NAME, {"recipe_name": "WorkflowRun", "success": False})

        assert mock_logger.log.call_count == 1
        assert mock_logger.log.call_args.args[0] == RECIPE_EVENT_NAME
    finally:
        Telemetry._instance = None


@pytest.mark.skipif(os.name != "nt", reason="Windows locking behavior is specific to Windows.")
def test_exclusive_file_lock_blocks_second_append_on_windows(tmp_path):
    file_path = tmp_path / "olive.json"
    child_code = """
import sys
import time
from pathlib import Path
from olive.telemetry.utils import _exclusive_file_lock

path = Path(sys.argv[1])
path.write_text("payload", encoding="utf-8")
with _exclusive_file_lock(path, "a") as locked_file:
    locked_file.write("child")
    locked_file.flush()
    print("locked", flush=True)
    time.sleep(2)
"""

    process = subprocess.Popen(
        [sys.executable, "-c", child_code, str(file_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"

        start = time.perf_counter()
        with _exclusive_file_lock(file_path, mode="a") as locked_file:
            wait_time = time.perf_counter() - start
            locked_file.write("parent")

        assert wait_time >= 1.0
    finally:
        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"child lock process timed out: stdout={stdout!r} stderr={stderr!r}")

    assert process.returncode == 0, stderr
    assert file_path.read_text(encoding="utf-8") == "payloadchildparent"
