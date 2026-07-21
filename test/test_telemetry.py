# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=duplicate-code,protected-access,redefined-outer-name
"""Tests for the SQLite-backed telemetry pipeline.

Covers the three-state opt-out semantics (CI / user opt-out / enabled), the
ALLOWED_KEYS whitelist filtering, the durable SQLite store, the single-drainer
process lock, the background uploader's success/poison/transient handling, and
the Common Schema serialization helpers. No test touches the network or the real
user profile: the HTTP transport is stubbed and the store directory is
redirected to a temp dir.
"""

import json
import os
import stat
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import olive.telemetry.deviceid._store as deviceid_store_mod
import olive.telemetry.library.transport as transport_mod
import olive.telemetry.telemetry as tmod
import olive.telemetry.utils as telemetry_utils
from olive.telemetry.library.connection_string_parser import ConnectionStringParser
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper as Serializer
from olive.telemetry.offline_store import SCHEMA_VERSION, OfflineEventStore
from olive.telemetry.process_lock import ProcessDrainLock
from olive.telemetry.uploader import EventUploader

ACTION_EVENT_NAME = tmod.ACTION_EVENT_NAME
ERROR_EVENT_NAME = tmod.ERROR_EVENT_NAME
HEARTBEAT_EVENT_NAME = tmod.HEARTBEAT_EVENT_NAME
RECIPE_EVENT_NAME = tmod.RECIPE_EVENT_NAME
Telemetry = tmod.Telemetry
is_ci_environment = tmod.is_ci_environment

_OPT_OUT_VAR = "ORT_DISABLE_TELEMETRY"
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
    monkeypatch.setattr(tmod, "get_telemetry_base_dir", lambda: tmp_path)
    monkeypatch.setattr(telemetry_utils, "get_telemetry_base_dir", lambda: tmp_path)
    monkeypatch.setattr(deviceid_store_mod, "get_telemetry_base_dir", lambda: tmp_path)

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
    """Join the heartbeat and drain the uploader.

    This makes recorded sends and store counts deterministic.
    """
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
        names.extend(
            token.decode()
            for token in (b"OliveHeartbeat", b"OliveRecipe", b"OliveAction", b"OliveError")
            if token in payload
        )
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
    assert t.accepts_detailed_events is False

    t.log(RECIPE_EVENT_NAME, {"recipe_name": "r", "success": True})
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" not in names
    assert "OliveRecipe" in names


def test_user_opt_out_records_heartbeat_only(tenv, monkeypatch):
    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    t = Telemetry()

    # Detailed events are not recorded or drained; the opt-out heartbeat is sent directly.
    assert t._enabled is False
    assert t._store is None
    assert t._uploader is None
    assert t._heartbeat_thread is not None
    assert t.accepts_detailed_events is False

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
    assert t.accepts_detailed_events is True

    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" in names
    assert "OliveAction" in names


def test_initialization_keeps_exporter_diagnostics_configurable(tenv):
    from olive.telemetry.library.event_source import event_source

    event_source.logger.disabled = False
    Telemetry()

    assert event_source.logger.disabled is False


def test_disable_telemetry_stops_detailed_events(tenv):
    t = Telemetry()
    _quiesce(t)
    t.disable_telemetry()

    assert t._enabled is False
    before = t._store.count() if t._store is not None else 0
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})
    after = t._store.count() if t._store is not None else 0
    assert after == before


def test_shutdown_joins_heartbeat_before_closing_store():
    t = object.__new__(Telemetry)
    t._heartbeat_thread = MagicMock()
    t._heartbeat_thread.is_alive.return_value = False
    t._uploader = None
    t._store = MagicMock()

    t.shutdown(callback_timeout_millis=250)

    assert t._heartbeat_thread is None
    assert t._store is None


def test_shutdown_uses_one_overall_budget():
    t = object.__new__(Telemetry)
    t._heartbeat_thread = MagicMock()
    t._heartbeat_thread.is_alive.return_value = False
    t._uploader = MagicMock()
    t._uploader.stop_loop.return_value = True
    t._store = MagicMock()
    heartbeat = t._heartbeat_thread
    uploader = t._uploader

    with patch("olive.telemetry.telemetry.time.monotonic", side_effect=[100.0, 101.0, 102.0, 103.0]):
        t.shutdown(timeout_millis=5_000, callback_timeout_millis=5_000, flush_seconds=5)

    heartbeat.join.assert_called_once_with(4.0)
    uploader.stop_loop.assert_called_once_with(join_timeout_seconds=3.0)
    uploader.flush.assert_called_once_with(2.0)
    assert t._heartbeat_thread is None
    assert t._uploader is None
    assert t._store is None


def test_closed_store_disables_telemetry(tenv):
    closed_store = MagicMock(is_open=False)
    with (
        patch.object(tmod, "OfflineEventStore", return_value=closed_store),
        patch.object(tmod, "EventUploader") as mock_uploader,
    ):
        t = Telemetry()

    assert t._enabled is False
    assert t._store is None
    assert t._heartbeat_thread is None
    mock_uploader.assert_not_called()


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


def test_event_attributes_override_metadata(tenv):
    t = Telemetry()
    _quiesce(t)
    payload = t._build_payload(
        ERROR_EVENT_NAME,
        {"exception_type": "ValueError", "exception_message": "safe"},
        {"exception_message": r"C:\Users\Mallory\secret.txt"},
    )
    assert json.loads(payload)["data"]["exception_message"] == "safe"


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
    db = os.path.join(tempfile.mkdtemp(), "olive_telemetry.db")
    return OfflineEventStore(db, **kwargs)


def test_store_is_fifo():
    store = _new_store()
    for i in range(5):
        store.store(f'{{"e":{i}}}'.encode())
    assert store.count() == 5
    batch = store.get_batch(3)
    assert [payload for _, payload in batch] == [b'{"e":0}', b'{"e":1}', b'{"e":2}']


def test_store_closes_connection_when_initialization_fails(tmp_path):
    connection = MagicMock()
    connection.execute.side_effect = RuntimeError("pragma failed")

    with patch("olive.telemetry.offline_store.sqlite3.connect", return_value=connection):
        store = OfflineEventStore(str(tmp_path / "failed.db"))

    assert store.is_open is False
    connection.close.assert_called_once()


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


@pytest.mark.skipif(os.name == "nt", reason="POSIX permissions")
def test_store_uses_owner_only_permissions():
    store = _new_store()
    db_path = Path(store.db_path)
    assert stat.S_IMODE(db_path.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(db_path.stat().st_mode) == 0o600


def test_empty_permission_path_does_not_chmod_cwd():
    from olive.telemetry.offline_store import _chmod_best_effort

    with patch("olive.telemetry.offline_store.Path.chmod") as mock_chmod:
        _chmod_best_effort("", 0o700)

    mock_chmod.assert_not_called()


# --------------------------------------------------------------------------
# Single-drainer process lock
# --------------------------------------------------------------------------


def _lock_path():
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
    assert Serializer.serialize_value({0: "zero", "": "skip"}) == {"0": "zero"}
    assert Serializer.serialize_value({False: "false"}) == {"False": "false"}


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
    with pytest.raises(ValueError, match="Connection string cannot be empty"):
        ConnectionStringParser("")
    with pytest.raises(ValueError, match="InstrumentationKey"):
        ConnectionStringParser("SomeOtherKey=value")


# --------------------------------------------------------------------------
# Exception-message path redaction (privacy)
# --------------------------------------------------------------------------


def test_redact_paths_matches_ort_scrubber():
    from olive.telemetry.telemetry_extensions import _redact_paths

    assert _redact_paths(r"C:\Users\alice\model.onnx") == "[path]"
    assert _redact_paths("/var/data/run/output.log") == "[path]"
    # Last segment is a directory/username (no extension) -> fully redacted.
    assert _redact_paths("/home/bob") == "[path]"
    # UNC paths are redacted too.
    assert _redact_paths(r"\\server\share\secret") == "[path]"
    assert _redact_paths(r"failed C:\Users\Alice Smith\models\phi.onnx") == "failed [path]"
    assert _redact_paths("failed /home/Alice Smith/models/phi.onnx") == "failed [path]"
    assert _redact_paths("a/b/c") == "[path]"
    assert _redact_paths(r"Load Users\bob\model.onnx failed") == "Load [path]"
    assert _redact_paths("models/foo.onnx") == "models/foo.onnx"
    assert _redact_paths("ratio 3/4 and and/or") == "ratio 3/4 and and/or"
    assert _redact_paths("before /home/alice/model.onnx\nafter") == "before [path]"
    assert len(_redact_paths("x" * 300).encode("utf-8")) == 256
    assert _redact_paths("x" * 255 + "€") == "x" * 255


def test_format_exception_message_redacts_paths_in_message():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    exc = RuntimeError(r"failed to read C:\Users\alice\secret\weights.bin")
    message = _format_exception_message(exc, exc.__traceback__)
    assert "alice" not in message
    assert "[path]" in message


def test_action_and_error_metadata_are_recursively_scrubbed():
    from olive.telemetry.telemetry_extensions import log_action, log_error

    telemetry = MagicMock()
    metadata = {
        "path": r"C:\Users\alice\models\model.onnx",
        r"C:\Users\alice\secret": "value",
        "nested": {
            "/home/alice/private/key": "value",
            "paths": ["/home/alice/model.onnx"],
        },
    }
    with patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry):
        log_action("test", "work", 1.0, True, metadata)
        action_metadata = telemetry.log.call_args.args[2]
        log_error("RuntimeError", "boom", metadata)
        error_metadata = telemetry.log.call_args.args[2]

    for scrubbed in (action_metadata, error_metadata):
        assert scrubbed["path"] == "[path]"
        assert scrubbed["nested"]["paths"] == ["[path]"]
        assert scrubbed["[path]"] == "value"
        assert scrubbed["nested"]["[path]"] == "value"


def test_format_exception_message_removes_external_path_cleanly():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    with patch(
        "olive.telemetry.telemetry_extensions.traceback.format_exception",
        return_value=['  File "/home/Alice Smith/project/external.py", line 12, in run\n'],
    ):
        message = _format_exception_message(RuntimeError("boom"))

    assert message == 'File "[path]", line 12, in run'


def test_format_exception_message_keeps_internal_basename_and_context():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    with patch(
        "olive.telemetry.telemetry_extensions.traceback.format_exception",
        return_value=['  File "/venv/site-packages/olive/telemetry/telemetry.py", line 9, in run\n'],
    ):
        message = _format_exception_message(RuntimeError("boom"))

    assert message == 'File "[path]", line 9, in run'


def test_device_id_store_uses_owner_only_creation_mode(tmp_path):
    with (
        patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path),
        patch.object(Path, "mkdir") as mock_mkdir,
    ):
        deviceid_store_mod.Store().store_id("test-device-id")

    mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)


def test_missing_device_id_raises_file_not_found(tmp_path):
    with (
        patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path),
        pytest.raises(FileNotFoundError),
    ):
        _ = deviceid_store_mod.Store().retrieve_id


def test_windows_device_id_store_uses_least_privilege_access():
    winreg = MagicMock(
        HKEY_CURRENT_USER=object(),
        KEY_SET_VALUE=0x0002,
        KEY_CREATE_SUB_KEY=0x0004,
        KEY_WOW64_64KEY=0x0100,
        REG_SZ=1,
    )
    key_handle = object()
    winreg.CreateKeyEx.return_value.__enter__.return_value = key_handle

    with patch.dict("sys.modules", {"winreg": winreg}):
        deviceid_store_mod.WindowsStore().store_id("test-device-id")

    winreg.CreateKeyEx.assert_called_once_with(
        winreg.HKEY_CURRENT_USER,
        deviceid_store_mod.REGISTRY_PATH,
        reserved=0,
        access=winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WOW64_64KEY,
    )
    winreg.SetValueEx.assert_called_once_with(
        key_handle,
        deviceid_store_mod.REGISTRY_KEY,
        0,
        winreg.REG_SZ,
        "test-device-id",
    )


def test_nested_actions_log_error_once():
    from olive.telemetry.telemetry_extensions import action

    telemetry = MagicMock(accepts_detailed_events=True)

    @action
    @action
    def fail():
        raise ValueError("boom")

    with (
        patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
        patch("olive.telemetry.telemetry_extensions.log_error") as mock_log_error,
        pytest.raises(ValueError, match="boom"),
    ):
        fail()

    mock_log_error.assert_called_once()


def test_positional_function_uses_function_action_name():
    from olive.telemetry.telemetry_extensions import action

    telemetry = MagicMock(accepts_detailed_events=True)

    @action
    def work(value):
        return value

    with (
        patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
        patch("olive.telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
        patch("olive.telemetry.telemetry_extensions.log_action") as mock_log_action,
    ):
        assert work("value") == "value"

    assert mock_log_action.call_args.kwargs["action_name"] == "work"


def test_action_context_without_start_time_reports_zero_duration():
    from olive.telemetry.telemetry_extensions import ActionContext

    telemetry = MagicMock(accepts_detailed_events=True)
    with (
        patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
        patch("olive.telemetry.telemetry_extensions._resolve_invoked_from", return_value="test"),
        patch("olive.telemetry.telemetry_extensions.time.perf_counter", return_value=100.0),
        patch("olive.telemetry.telemetry_extensions.log_action") as mock_log_action,
    ):
        context = ActionContext("work")
        context.__exit__(None, None, None)

    assert mock_log_action.call_args.kwargs["duration_ms"] == 0


def test_disabled_action_skips_stack_inspection():
    from olive.telemetry.telemetry_extensions import action

    telemetry = MagicMock(accepts_detailed_events=False)

    @action
    def work():
        return 42

    with (
        patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
        patch("olive.telemetry.telemetry_extensions._resolve_invoked_from") as mock_resolve,
    ):
        assert work() == 42

    mock_resolve.assert_not_called()


def test_disabled_action_context_skips_stack_inspection():
    from olive.telemetry.telemetry_extensions import ActionContext

    telemetry = MagicMock(accepts_detailed_events=False)
    with (
        patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry),
        patch("olive.telemetry.telemetry_extensions._resolve_invoked_from") as mock_resolve,
        patch("olive.telemetry.telemetry_extensions.log_action") as mock_log_action,
        ActionContext("work"),
    ):
        result = 42

    assert result == 42
    mock_resolve.assert_not_called()
    mock_log_action.assert_not_called()
