# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""Tests for the SQLite-backed telemetry pipeline.

Covers the three-state opt-out semantics (CI / user opt-out / enabled), the
ALLOWED_KEYS whitelist filtering, the durable SQLite store, the single-drainer
process lock, the background uploader's success/poison/transient handling, and
the Common Schema serialization helpers. No test touches the network or the real
user profile: the HTTP transport is stubbed and the store directory is
redirected to a temp dir.
"""

import json
import tempfile
from types import SimpleNamespace

import pytest

import olive.telemetry.library.transport as transport_mod
import olive.telemetry.telemetry as tmod
from olive.telemetry.library.connection_string_parser import ConnectionStringParser
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper as Serializer
from olive.telemetry.offline_store import SCHEMA_VERSION, OfflineEventStore
from olive.telemetry.process_lock import ProcessDrainLock
from olive.telemetry.telemetry import (
    ACTION_EVENT_NAME,
    ERROR_EVENT_NAME,
    HEARTBEAT_EVENT_NAME,
    RECIPE_EVENT_NAME,
    Telemetry,
    is_ci_environment,
)
from olive.telemetry.uploader import EventUploader

_OPT_OUT_VAR = "OLIVE_DISABLE_TELEMETRY"
_CI_VARS = (
    "CI",
    "TF_BUILD",
    "GITHUB_ACTIONS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
)


@pytest.fixture
def tenv(tmp_path, monkeypatch):
    """Hermetic telemetry environment.

    Clears CI/opt-out signals so each test sets its own mode, stubs the HTTP
    transport (recording every send in ``.sends``), and redirects the durable
    store off the real profile. On teardown the heartbeat thread is joined
    BEFORE monkeypatch restores the real transport, so a lagging heartbeat can
    never POST real device data from a test.
    """
    Telemetry._instance = None
    for var in (_OPT_OUT_VAR, *_CI_VARS):
        monkeypatch.delenv(var, raising=False)

    sends = []

    def _record_send(self, payload, timeout_sec, item_count=1):
        sends.append({"item_count": item_count, "size": len(payload), "payload": payload})
        return True, 204

    monkeypatch.setattr(transport_mod.HttpJsonPostTransport, "send", _record_send)
    monkeypatch.setattr(tmod, "get_telemetry_base_dir", lambda: str(tmp_path))

    yield SimpleNamespace(sends=sends, tmp_path=tmp_path)

    inst = Telemetry._instance
    if inst is not None:
        uploader = getattr(inst, "_uploader", None)
        if uploader is not None:
            uploader.stop_loop(5)
        heartbeat = getattr(inst, "_heartbeat_thread", None)
        if heartbeat is not None:
            heartbeat.join()
    Telemetry._instance = None


def _quiesce(t):
    """Join the heartbeat (so it is enqueued) and drain the uploader so the
    recorded sends and store counts are deterministic."""
    heartbeat = getattr(t, "_heartbeat_thread", None)
    if heartbeat is not None:
        heartbeat.join()
    if t._uploader is not None:
        t._uploader.stop_loop(5)
        for _ in range(20):
            if t._store is None or t._store.count() == 0:
                break
            t._uploader.drain_once()


def _sent_event_names(sends):
    names = []
    for s in sends:
        payload = bytes(s["payload"])
        for token in (b"OliveHeartbeat", b"OliveRecipe", b"OliveAction", b"OliveError"):
            if token in payload:
                names.append(token.decode())
    return names


# --------------------------------------------------------------------------
# Three-state opt-out semantics
# --------------------------------------------------------------------------


def test_ci_is_recipe_only_with_no_heartbeat(tenv, monkeypatch):
    monkeypatch.setenv("CI", "1")
    t = Telemetry()

    # CI suppresses the device-id heartbeat but still persists recipe events.
    assert t._heartbeat_thread is None
    assert t._store is not None

    before = t._store.count()
    t.log(RECIPE_EVENT_NAME, {"recipe_name": "r", "success": True})
    assert t._store.count() == before + 1

    middle = t._store.count()
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})
    assert t._store.count() == middle  # non-recipe events suppressed in CI

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" not in names
    assert "OliveRecipe" in names


def test_user_opt_out_records_heartbeat_only(tenv, monkeypatch):
    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    t = Telemetry()

    # Detailed events are not recorded, but the heartbeat is durably queued.
    assert t._enabled is False
    assert t._store is not None
    assert t._heartbeat_thread is not None

    # Detailed-event methods are no-ops and must not raise.
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" in names
    assert "OliveAction" not in names


def test_opt_out_and_ci_send_nothing(tenv, monkeypatch):
    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    monkeypatch.setenv("CI", "1")
    t = Telemetry()
    _quiesce(t)

    # Explicit opt-out + CI: record and send nothing at all.
    assert t._enabled is False
    assert t._store is None
    assert t._heartbeat_thread is None
    assert tenv.sends == []


def test_enabled_records_heartbeat_and_events(tenv):
    t = Telemetry()

    assert t._enabled is True
    assert t._store is not None

    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" in names
    assert "OliveAction" in names


def test_disable_telemetry_stops_detailed_events(tenv):
    t = Telemetry()
    _quiesce(t)
    t.disable_telemetry()

    assert t._enabled is False
    before = t._store.count() if t._store is not None else 0
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})
    after = t._store.count() if t._store is not None else 0
    assert after == before


# --------------------------------------------------------------------------
# Whitelist filtering / payload building
# --------------------------------------------------------------------------


def test_build_payload_drops_non_whitelisted_keys(tenv):
    t = Telemetry()
    _quiesce(t)

    payload = t._build_payload(
        ACTION_EVENT_NAME,
        {
            "invoked_from": "cli",
            "action_name": "WorkflowRun",
            "duration_ms": 1.0,
            "success": True,
            "secret": "SHOULD_NOT_BE_SENT",
        },
    )
    data = json.loads(payload)["data"]
    assert "secret" not in data
    assert data["action_name"] == "WorkflowRun"
    # Defaults are stamped on every event.
    assert data["app_version"]
    assert data["app_instance_id"]


def test_build_payload_returns_none_for_unknown_event(tenv):
    t = Telemetry()
    _quiesce(t)
    assert t._build_payload("TotallyUnknownEvent", {"k": "v"}) is None


def test_build_payload_heartbeat_uses_flat_os_fields(tenv):
    t = Telemetry()
    _quiesce(t)

    payload = t._build_payload(
        HEARTBEAT_EVENT_NAME,
        {
            "device_id": "DEVICE",
            "device_id_status": "ok",
            "os": "Windows",
            "os_version": "10.0.22631",
            "os_release": "11",
            "os_arch": "AMD64",
            "leak": "DROP",
        },
    )
    data = json.loads(payload)["data"]
    assert data["device_id"] == "DEVICE"
    assert data["device_id_status"] == "ok"
    assert data["os"] == "Windows"
    assert data["os_version"] == "10.0.22631"
    assert "leak" not in data


def test_global_metadata_is_merged_then_filtered(tenv):
    t = Telemetry()
    _quiesce(t)

    # app_version is whitelisted for actions; not_allowed is not.
    t.add_global_metadata({"app_version": "9.9.9", "not_allowed": "DROP"})
    payload = t._build_payload(
        ACTION_EVENT_NAME,
        {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True},
    )
    data = json.loads(payload)["data"]
    assert data["app_version"] == "9.9.9"
    assert "not_allowed" not in data


def test_error_event_whitelist(tenv):
    t = Telemetry()
    _quiesce(t)
    payload = t._build_payload(
        ERROR_EVENT_NAME,
        {"exception_type": "RuntimeError", "exception_message": "boom", "stack": "SENSITIVE"},
    )
    data = json.loads(payload)["data"]
    assert data["exception_type"] == "RuntimeError"
    assert data["exception_message"] == "boom"
    assert "stack" not in data


# --------------------------------------------------------------------------
# CI detection
# --------------------------------------------------------------------------


def test_is_ci_environment(monkeypatch):
    for var in (_OPT_OUT_VAR, *_CI_VARS):
        monkeypatch.delenv(var, raising=False)
    assert is_ci_environment() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert is_ci_environment() is True


# --------------------------------------------------------------------------
# Durable SQLite store
# --------------------------------------------------------------------------


def _new_store(**kwargs):
    import os

    db = os.path.join(tempfile.mkdtemp(), "olive_telemetry.db")
    return OfflineEventStore(db, **kwargs)


def test_store_is_fifo():
    store = _new_store()
    for i in range(5):
        store.store(f'{{"e":{i}}}'.encode())
    assert store.count() == 5
    batch = store.get_batch(3)
    assert [payload for _, payload in batch] == [b'{"e":0}', b'{"e":1}', b'{"e":2}']


def test_store_delete():
    store = _new_store()
    store.store(b'{"a":1}')
    store.store(b'{"b":2}')
    ids = [row_id for row_id, _ in store.get_batch(10)]
    store.delete(ids[:1])
    assert store.count() == 1


def test_store_trims_over_capacity():
    store = _new_store(max_records=8)
    for i in range(40):
        store.store(f'{{"i":{i}}}'.encode())
    assert store.count() <= 8


def test_store_rejects_empty_payload():
    store = _new_store()
    assert store.store(b"") is False


def test_store_stamps_schema_version():
    import sqlite3

    store = _new_store()
    version = sqlite3.connect(store.db_path).execute("PRAGMA user_version").fetchone()[0]
    assert version == SCHEMA_VERSION


# --------------------------------------------------------------------------
# Single-drainer process lock
# --------------------------------------------------------------------------


def _lock_path():
    import os

    return os.path.join(tempfile.mkdtemp(), "olive_telemetry.db.lock")


def test_lock_is_mutually_exclusive():
    path = _lock_path()
    a = ProcessDrainLock(path)
    b = ProcessDrainLock(path)
    assert a.acquire() is True
    assert b.acquire() is False  # held by a
    a.release()
    assert b.acquire() is True  # released
    b.release()


def test_lock_reacquire_is_idempotent():
    a = ProcessDrainLock(_lock_path())
    assert a.acquire() is True
    assert a.acquire() is True  # already held
    assert a.held is True
    a.release()
    assert a.held is False


# --------------------------------------------------------------------------
# Uploader drain classification (no real network)
# --------------------------------------------------------------------------


def _store_and_uploader():
    import os

    db = os.path.join(tempfile.mkdtemp(), "olive_telemetry.db")
    store = OfflineEventStore(db)
    uploader = EventUploader(store, instrumentation_key="abc-def")
    return store, uploader


def test_uploader_deletes_on_success():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = lambda *a, **k: (True, 204)
    delivered, left = uploader.drain_once()
    assert (delivered, left) == (1, 0)
    assert store.count() == 0


def test_uploader_drops_poison_4xx():
    store, uploader = _store_and_uploader()
    store.store(b'{"bad":1}')
    uploader._transport.send = lambda *a, **k: (False, 400)
    uploader.drain_once()
    assert store.count() == 0  # dropped, not retried forever


def test_uploader_retains_transient_5xx():
    store, uploader = _store_and_uploader()
    store.store(b'{"later":1}')
    uploader._transport.send = lambda *a, **k: (False, 503)
    delivered, left = uploader.drain_once()
    assert (delivered, left) == (0, 1)
    assert store.count() == 1  # kept for retry


# --------------------------------------------------------------------------
# Serialization + connection string parsing
# --------------------------------------------------------------------------


def test_serialize_basic_types():
    assert Serializer.serialize_value(None) is None
    assert Serializer.serialize_value(True) is True
    assert Serializer.serialize_value(42) == 42
    assert Serializer.serialize_value("hello") == "hello"
    assert Serializer.serialize_value([1, "two", 3.0]) == [1, "two", 3.0]
    assert Serializer.serialize_value({"k": "v"}) == {"k": "v"}


def test_create_event_envelope():
    from datetime import datetime, timezone

    envelope = Serializer.create_event_envelope(
        event_name="TestEvent",
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        ikey="o:test-key",
        data={"key": "value"},
    )
    assert envelope["name"] == "TestEvent"
    assert envelope["iKey"] == "o:test-key"
    assert envelope["data"] == {"key": "value"}


def test_connection_string_parser():
    assert ConnectionStringParser("InstrumentationKey=abc-def-ghi").instrumentation_key == "abc-def-ghi"
    with pytest.raises(ValueError):
        ConnectionStringParser("")
    with pytest.raises(ValueError):
        ConnectionStringParser("SomeOtherKey=value")


# --------------------------------------------------------------------------
# Exception-message path redaction (privacy)
# --------------------------------------------------------------------------


def test_redact_paths_keeps_filenames_drops_usernames():
    from olive.telemetry.telemetry_extensions import _redact_paths

    assert _redact_paths(r"C:\Users\alice\model.onnx") == "model.onnx"
    assert _redact_paths("/var/data/run/output.log") == "output.log"
    # Last segment is a directory/username (no extension) -> fully redacted.
    assert _redact_paths("/home/bob") == "<path>"
    # UNC paths are redacted too.
    assert _redact_paths(r"\\server\share\secret") == "<path>"


def test_format_exception_message_redacts_paths_in_message():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    try:
        raise RuntimeError(r"failed to read C:\Users\alice\secret\weights.bin")
    except RuntimeError as exc:
        message = _format_exception_message(exc, exc.__traceback__)
    assert "alice" not in message
    assert "weights.bin" in message
