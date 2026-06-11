# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

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


def test_cache_path_ignores_empty_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("OLIVE_TELEMETRY_CACHE_DIR", "   ")

    with patch("olive.telemetry.telemetry.get_telemetry_base_dir", return_value=tmp_path):
        handler = TelemetryCacheHandler(Mock())
        assert handler.cache_path == tmp_path / "cache" / CACHE_FILE_NAME


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
        assert telemetry._cache_handler is None
        mock_logger.register_payload_transmitted_callback.assert_not_called()
    finally:
        Telemetry._instance = None


def test_flush_cache_preserves_nonempty_unreadable_file(tmp_path):
    handler = TelemetryCacheHandler(Mock())
    cache_path = tmp_path / CACHE_FILE_NAME
    flush_path = cache_path.with_name(f"{cache_path.name}.flush")
    cache_path.write_text("not-json\n", encoding="utf-8")

    handler._flush_cache_file(cache_path)

    assert cache_path.exists()
    assert cache_path.read_text(encoding="utf-8") == "not-json\n"
    assert not flush_path.exists()


def _write_cache_entry(cache_path, event_name="TestEvent", payload=None):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "event_name": event_name,
        "event_data": json.dumps(payload if payload is not None else {"key": "value"}),
        "ts": 12345,
        "initTs": 12345,
    }
    cache_path.write_text(json.dumps(entry) + "\n", encoding="utf-8")
    return entry


def _make_replay_handler(success):
    telemetry = Mock()
    handler = TelemetryCacheHandler(telemetry)
    # Pretend we're already in a flush so callbacks are treated as replays.
    handler._is_flushing = True

    def fake_log(_event_name, _attrs, _metadata):
        handler.record_event_logged()
        handler.on_payload_transmitted(SimpleNamespace(succeeded=success, item_count=1, payload_bytes=b""))

    telemetry.log.side_effect = fake_log
    return handler, telemetry


def test_flush_deletes_cache_when_replay_succeeds(tmp_path):
    handler, _ = _make_replay_handler(success=True)
    cache_path = tmp_path / CACHE_FILE_NAME
    flush_path = cache_path.with_name(f"{cache_path.name}.flush")
    _write_cache_entry(cache_path)

    handler._flush_cache_file(cache_path)

    assert not cache_path.exists()
    assert not flush_path.exists()


def test_flush_restores_cache_when_replay_fails(tmp_path):
    handler, _ = _make_replay_handler(success=False)
    cache_path = tmp_path / CACHE_FILE_NAME
    flush_path = cache_path.with_name(f"{cache_path.name}.flush")
    _write_cache_entry(cache_path, event_name="ReplayedEvent")

    handler._flush_cache_file(cache_path)

    # Failed replay must preserve the cached event so a later flush can retry,
    # rather than silently dropping it.
    assert cache_path.exists()
    assert "ReplayedEvent" in cache_path.read_text(encoding="utf-8")
    assert not flush_path.exists()


def test_flush_restores_cache_when_callbacks_timeout(tmp_path, monkeypatch):
    telemetry = Mock()
    handler = TelemetryCacheHandler(telemetry)
    handler._is_flushing = True
    cache_path = tmp_path / CACHE_FILE_NAME
    flush_path = cache_path.with_name(f"{cache_path.name}.flush")
    _write_cache_entry(cache_path, event_name="OrphanedEvent")

    # Simulate replay that logs the event but never fires the callback
    # (e.g. exporter dropped or stalled). wait_for_callbacks should time out.
    def fake_log(_event_name, _attrs, _metadata):
        handler.record_event_logged()

    telemetry.log.side_effect = fake_log
    monkeypatch.setattr(handler, "wait_for_callbacks", lambda **_: False)

    handler._flush_cache_file(cache_path)

    assert cache_path.exists()
    assert "OrphanedEvent" in cache_path.read_text(encoding="utf-8")
    assert not flush_path.exists()


def test_wait_until_flush_complete_wakes_when_flush_clears():
    handler = TelemetryCacheHandler(Mock())
    handler._is_flushing = True

    def clear_flag():
        time.sleep(0.05)
        with handler._condition:
            handler._is_flushing = False
            handler._condition.notify_all()

    threading.Thread(target=clear_flag, daemon=True).start()

    start = time.perf_counter()
    completed = handler.wait_until_flush_complete(1.0)
    elapsed = time.perf_counter() - start

    assert completed is True
    # Should wake on notify, not poll the full timeout
    assert elapsed < 0.5


def test_wait_until_flush_complete_returns_false_on_timeout():
    handler = TelemetryCacheHandler(Mock())
    handler._is_flushing = True

    assert handler.wait_until_flush_complete(0.05) is False


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

    with subprocess.Popen(
        [sys.executable, "-c", child_code, str(file_path)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"

        start = time.perf_counter()
        with _exclusive_file_lock(file_path, mode="a") as locked_file:
            wait_time = time.perf_counter() - start
            locked_file.write("parent")

        assert wait_time >= 1.0

        try:
            stdout, stderr = process.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            pytest.fail(f"child lock process timed out: stdout={stdout!r} stderr={stderr!r}")

    assert process.returncode == 0, stderr
    assert file_path.read_text(encoding="utf-8") == "payloadchildparent"
