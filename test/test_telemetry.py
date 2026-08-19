# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=duplicate-code,protected-access,redefined-outer-name
"""Tests for the SQLite-backed telemetry pipeline.

Covers enabled, full opt-out, and CI recipe-only semantics; the
ALLOWED_KEYS whitelist filtering, the durable SQLite store, the single-drainer
process lock, the background uploader's success/poison/transient handling, and
the Common Schema serialization helpers. No test touches the network or the real
user profile: the HTTP transport is stubbed and the store directory is
redirected to a temp dir.
"""

import hashlib
import json
import os
import sqlite3
import stat
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

import olive.telemetry.deviceid._store as deviceid_store_mod
import olive.telemetry.library.transport as transport_mod
import olive.telemetry.telemetry as tmod
import olive.telemetry.utils as telemetry_utils
from olive.telemetry.library.connection_string_parser import ConnectionStringParser
from olive.telemetry.library.options import OneCollectorTransportOptions
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper as Serializer
from olive.telemetry.offline_store import OfflineEventStore
from olive.telemetry.process_lock import ProcessDrainLock
from olive.telemetry.telemetry_redaction import scrub_value_for_telemetry
from olive.telemetry.uploader import DrainOutcome, DrainResult, EventUploader

ACTION_EVENT_NAME = tmod.ACTION_EVENT_NAME
ERROR_EVENT_NAME = tmod.ERROR_EVENT_NAME
HEARTBEAT_EVENT_NAME = tmod.HEARTBEAT_EVENT_NAME
RECIPE_EVENT_NAME = tmod.RECIPE_EVENT_NAME
Telemetry = tmod.Telemetry
is_ci_environment = tmod.is_ci_environment

_FULL_DISABLE_VAR = "ORT_DISABLE_TELEMETRY"
_OPT_OUT_VAR = "OLIVE_DISABLE_TELEMETRY"
_CI_VARS = (
    "CI",
    "TF_BUILD",
    "GITHUB_ACTIONS",
    "GITLAB_CI",
    "CIRCLECI",
    "TRAVIS",
    "JENKINS_URL",
    "CODEBUILD_BUILD_ID",
    "BUILDKITE",
    "TEAMCITY_VERSION",
    "APPVEYOR",
    "BITBUCKET_BUILD_NUMBER",
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",
)


@pytest.fixture
def tenv(tmp_path, monkeypatch):
    """Hermetic telemetry environment.

    Clears CI/opt-out signals so each test sets its own mode, stubs the HTTP
    transport (recording every send in ``.sends``), and redirects the durable
    store off the real profile.
    """
    Telemetry._instance = None
    Telemetry._process_disabled = False
    Telemetry._heartbeat_enqueued = False
    for var in (_FULL_DISABLE_VAR, _OPT_OUT_VAR, *_CI_VARS):
        monkeypatch.delenv(var, raising=False)

    sends = []

    def _record_send(self, payload, timeout_sec, item_count=1, on_send_admitted=None):
        if on_send_admitted is not None:
            on_send_admitted()
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
    Telemetry._instance = None
    Telemetry._process_disabled = False
    Telemetry._heartbeat_enqueued = False


def _quiesce(t):
    """Stop the background loop and drain its durable queue deterministically."""
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
# Full opt-out semantics
# --------------------------------------------------------------------------


@pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on", "y"])
@pytest.mark.parametrize("variable", [_FULL_DISABLE_VAR, _OPT_OUT_VAR])
def test_environment_opt_out_uses_canonical_allowlist(tenv, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    assert tmod.is_telemetry_disabled_by_environment() is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "unexpected"])
@pytest.mark.parametrize("variable", [_FULL_DISABLE_VAR, _OPT_OUT_VAR])
def test_environment_opt_out_rejects_non_allowlisted_values(tenv, monkeypatch, variable, value):
    monkeypatch.setenv(variable, value)

    assert tmod.is_telemetry_disabled_by_environment() is False


def test_ci_is_recipe_only_with_no_heartbeat(tenv, monkeypatch):
    monkeypatch.setenv("CI", "1")
    t = Telemetry()

    # CI suppresses the device-id heartbeat but still persists recipe events.
    assert t._store is not None
    assert t.accepts_detailed_events is False
    assert t._uploader._thread is None

    t.log(RECIPE_EVENT_NAME, {"recipe_name": "r", "success": True})
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" not in names
    assert "OliveRecipe" in names


def test_ci_recipe_queue_does_not_drain_local_details(tenv, monkeypatch):
    local_store = OfflineEventStore(str(tenv.tmp_path / tmod.DB_FILE_NAME))
    local_store.store(b'{"name":"OliveAction"}')
    monkeypatch.setenv("CI", "1")

    telemetry = Telemetry()
    telemetry.log(RECIPE_EVENT_NAME, {"recipe_name": "r", "success": True})
    _quiesce(telemetry)

    assert local_store.count() == 1
    assert "OliveAction" not in _sent_event_names(tenv.sends)
    assert "OliveRecipe" in _sent_event_names(tenv.sends)
    local_store.close()


def test_ci_shutdown_flushes_recipe_within_callback_budget(tenv, monkeypatch):
    monkeypatch.setenv("CI", "1")
    with patch.object(EventUploader, "start") as start:
        telemetry = Telemetry()
        telemetry.log(RECIPE_EVENT_NAME, {"recipe_name": "r", "success": True})

    telemetry.shutdown(timeout_millis=0, callback_timeout_millis=1_000)

    start.assert_not_called()
    assert _sent_event_names(tenv.sends) == ["OliveRecipe"]


def test_ci_user_opt_out_fully_disables_telemetry(tenv, monkeypatch):
    monkeypatch.setenv("CI", "1")
    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    with (
        patch.object(tmod, "OfflineEventStore") as mock_store,
        patch.object(tmod, "get_hashed_device_id_and_status") as mock_device_id,
    ):
        telemetry = Telemetry()

    assert Telemetry._instance is None
    assert telemetry._enabled is False
    assert telemetry._store is None
    mock_store.assert_not_called()
    mock_device_id.assert_not_called()
    assert tenv.sends == []


def test_user_opt_out_sends_and_persists_nothing(tenv, monkeypatch):
    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    with (
        patch.object(tmod, "OfflineEventStore") as mock_store,
        patch.object(tmod, "EventUploader") as mock_uploader,
        patch.object(tmod, "get_hashed_device_id_and_status") as mock_device_id,
    ):
        t = Telemetry()

    assert t._enabled is False
    assert t._store is None
    assert t._uploader is None
    assert t.accepts_detailed_events is False
    mock_store.assert_not_called()
    mock_uploader.assert_not_called()
    mock_device_id.assert_not_called()
    assert tenv.sends == []


def test_user_opt_out_is_latched_across_reinitialization(tenv, monkeypatch):
    monkeypatch.setenv(_OPT_OUT_VAR, "true")
    first = Telemetry()
    first.shutdown()
    monkeypatch.delenv(_OPT_OUT_VAR)

    second = Telemetry()

    assert Telemetry._instance is None
    assert second is not first
    assert second._enabled is False
    assert second._store is None
    assert second._uploader is None
    assert tenv.sends == []


def test_full_disable_and_ci_send_nothing(tenv, monkeypatch):
    monkeypatch.setenv(_FULL_DISABLE_VAR, "1")
    monkeypatch.setenv("CI", "1")
    t = Telemetry()
    _quiesce(t)

    # Explicit full suppression + CI: record and send nothing at all.
    assert Telemetry._instance is None
    assert t._enabled is False
    assert t._store is None
    assert tenv.sends == []


def test_full_disable_sends_nothing(tenv, monkeypatch):
    monkeypatch.setenv(_FULL_DISABLE_VAR, "1")
    t = Telemetry()
    _quiesce(t)

    assert Telemetry._instance is None
    assert t._enabled is False
    assert t._store is None
    assert t._uploader is None
    assert tenv.sends == []


def test_public_disable_before_initialization_only_latches_suppression(tenv):
    from olive.telemetry.telemetry_extensions import log_action, log_error, log_recipe_result

    with (
        patch.object(tmod, "OfflineEventStore") as mock_store,
        patch.object(tmod, "EventUploader") as mock_uploader,
        patch.object(tmod, "get_hashed_device_id_and_status") as mock_device_id,
    ):
        tmod.disable_telemetry()
        assert Telemetry._instance is None
        log_action("cli", "work", 1.0, True)
        log_error("RuntimeError", "boom")
        log_recipe_result("recipe", True)
        assert Telemetry._instance is None
        telemetry = Telemetry()
        tmod.disable_telemetry()

    assert Telemetry._instance is None
    mock_store.assert_not_called()
    mock_uploader.assert_not_called()
    mock_device_id.assert_not_called()
    assert telemetry._enabled is False
    assert telemetry._store is None
    assert tenv.sends == []


def test_environment_opt_out_helpers_do_not_publish_singleton(tenv, monkeypatch):
    from olive.telemetry.telemetry_extensions import log_action, log_error, log_recipe_result

    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    log_action("cli", "work", 1.0, True)
    log_error("RuntimeError", "boom")
    log_recipe_result("recipe", True)

    assert Telemetry._instance is None


def test_environment_opt_out_after_initialization_retains_queue(tenv, monkeypatch):
    with patch.object(EventUploader, "start"):
        telemetry = Telemetry()
        telemetry.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1, "success": True})
        row_count = telemetry._store.count()

    monkeypatch.setenv(_OPT_OUT_VAR, "1")
    assert Telemetry.get_or_create_if_enabled() is None

    assert telemetry._disabled is True
    remaining_store = OfflineEventStore(str(tenv.tmp_path / tmod.DB_FILE_NAME))
    assert remaining_store.count() == row_count
    remaining_store.close()


def test_enabled_records_heartbeat_and_events(tenv):
    import uuid

    t = Telemetry()
    session_guid = uuid.UUID(t._app_session_guid)
    assert session_guid.version == 4
    assert session_guid.variant == uuid.RFC_4122

    assert t._enabled is True
    assert t._store is not None
    assert t.accepts_detailed_events is True

    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})

    _quiesce(t)
    names = _sent_event_names(tenv.sends)
    assert "OliveHeartbeat" in names
    assert "OliveAction" in names


def test_heartbeat_is_stored_durably_before_uploader_starts(tenv):
    with patch.object(EventUploader, "start"):
        telemetry = Telemetry()

    batch = telemetry._store.get_batch(10)
    assert len(batch) == 1
    assert json.loads(batch[0][1])["name"] == HEARTBEAT_EVENT_NAME
    assert tenv.sends == []


def test_heartbeat_releases_minimal_fallback_when_enrichment_fails(tenv):
    with (
        patch.object(EventUploader, "start"),
        patch.object(
            tmod,
            "get_hashed_device_id_and_status",
            return_value=("c:device", SimpleNamespace(value="Existing")),
        ),
        patch.object(tmod.platform, "system", side_effect=RuntimeError("unavailable")),
    ):
        telemetry = Telemetry()

    batch = telemetry._store.get_batch(10)
    assert len(batch) == 1
    data = json.loads(batch[0][1])["data"]
    assert data["deviceId"] == "c:device"
    assert data["deviceIdStatus"] == "Existing"
    assert "os" not in data


def test_disable_telemetry_stops_detailed_events(tenv):
    t = Telemetry()
    _quiesce(t)
    t.disable_telemetry()

    assert t._enabled is False
    before = t._store.count() if t._store is not None else 0
    t.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})
    after = t._store.count() if t._store is not None else 0
    assert after == before
    t.shutdown()
    assert Telemetry()._enabled is False
    assert _sent_event_names(tenv.sends).count("OliveHeartbeat") == 1


def test_runtime_disable_does_not_emit_an_additional_heartbeat(tenv):
    telemetry = Telemetry()
    _quiesce(telemetry)
    tenv.sends.clear()
    telemetry.disable_telemetry()
    telemetry.disable_telemetry()

    assert tenv.sends == []


def test_runtime_disable_retains_all_unsent_rows(tenv):
    with patch.object(EventUploader, "start"):
        telemetry = Telemetry()
        other_store = OfflineEventStore(telemetry._store.db_path)
        other_store.store(b'{"other":1}')
        telemetry.log(
            ACTION_EVENT_NAME,
            {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True},
        )

        telemetry.disable_telemetry()

    payloads = [payload for _, payload in other_store.get_batch(10)]
    assert b'{"other":1}' in payloads
    assert {json.loads(payload)["name"] for payload in payloads if payload != b'{"other":1}'} == {
        HEARTBEAT_EVENT_NAME,
        ACTION_EVENT_NAME,
    }
    other_store.close()


def test_runtime_disable_serializes_with_inflight_log(tenv):
    with patch.object(EventUploader, "start"):
        telemetry = Telemetry()
    entered = threading.Event()
    release = threading.Event()
    original_store = telemetry._store.store

    def blocked_store(payload):
        entered.set()
        assert release.wait(5)
        return original_store(payload)

    with patch.object(telemetry._store, "store", side_effect=blocked_store):
        logging_thread = threading.Thread(
            target=telemetry.log,
            args=(
                ACTION_EVENT_NAME,
                {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True},
            ),
        )
        logging_thread.start()
        assert entered.wait(5)
        disable_thread = threading.Thread(target=telemetry.disable_telemetry)
        disable_thread.start()
        release.set()
        logging_thread.join(5)
        disable_thread.join(5)

    assert not logging_thread.is_alive()
    assert not disable_thread.is_alive()
    assert OfflineEventStore(str(tenv.tmp_path / tmod.DB_FILE_NAME)).count() == 2


def test_shutdown_closes_store_without_uploader():
    t = object.__new__(Telemetry)
    t._disabled = False
    t._uploader = None
    t._store = MagicMock()
    t._initialized = True

    store = t._store
    t.shutdown()

    store.close.assert_called_once()
    assert t._store is None
    assert t._initialized is False


def test_shutdown_uses_one_overall_budget():
    t = object.__new__(Telemetry)
    t._disabled = False
    t._uploader = MagicMock()
    t._uploader.stop_loop.return_value = True
    t._store = MagicMock()
    uploader = t._uploader

    monotonic = MagicMock(side_effect=[100.0, 101.0, 102.0])
    with patch("olive.telemetry.telemetry.time", SimpleNamespace(monotonic=monotonic)):
        t.shutdown(timeout_millis=5_000, callback_timeout_millis=5_000, flush_seconds=5)

    uploader.stop_loop.assert_called_once_with(join_timeout_seconds=4.0)
    uploader.flush.assert_called_once_with(3.0)
    assert t._uploader is None
    assert t._store is None
    assert t._initialized is False


def test_shutdown_does_not_wait_or_flush_after_full_disable():
    telemetry = object.__new__(Telemetry)
    telemetry._initialized = True
    telemetry._disabled = True
    telemetry._uploader = MagicMock()
    telemetry._uploader.stop_loop.return_value = True
    telemetry._store = MagicMock()
    uploader = telemetry._uploader

    telemetry.shutdown(timeout_millis=5_000, callback_timeout_millis=5_000, flush_seconds=5)

    uploader.stop_loop.assert_called_once_with(join_timeout_seconds=0)
    uploader.flush.assert_not_called()


def test_reinitialization_does_not_enqueue_second_heartbeat(tenv):
    telemetry = Telemetry()
    _quiesce(telemetry)
    telemetry.shutdown()

    restarted = Telemetry()
    restarted.log(ACTION_EVENT_NAME, {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True})
    _quiesce(restarted)

    names = _sent_event_names(tenv.sends)
    assert names.count("OliveHeartbeat") == 1
    assert names.count("OliveAction") == 1


def test_live_uploader_keeps_store_open():
    telemetry = object.__new__(Telemetry)
    telemetry._disabled = False
    telemetry._uploader = MagicMock()
    telemetry._uploader.stop_loop.return_value = False
    telemetry._store = MagicMock()

    telemetry.shutdown(timeout_millis=0)

    telemetry._store.close.assert_not_called()


def test_closed_store_disables_telemetry(tenv):
    closed_store = MagicMock(is_open=False)
    with (
        patch.object(tmod, "OfflineEventStore", return_value=closed_store),
        patch.object(tmod, "EventUploader") as mock_uploader,
    ):
        t = Telemetry()

    assert t._enabled is False
    assert t._store is None
    assert Telemetry._heartbeat_enqueued is False
    mock_uploader.assert_not_called()


def test_closed_store_allows_initialization_retry(tenv):
    closed_store = MagicMock(is_open=False)
    open_store = MagicMock(is_open=True)
    with (
        patch.object(tmod, "OfflineEventStore", side_effect=[closed_store, open_store]),
        patch.object(tmod, "EventUploader") as mock_uploader,
    ):
        first = Telemetry()
        assert first._initialized is False

        second = Telemetry()

    assert second is first
    assert second._initialized is True
    assert second._enabled is True
    mock_uploader.assert_called_once_with(open_store, instrumentation_key=second._instrumentation_key)


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
    assert data["actionName"] == "WorkflowRun"
    # Defaults are stamped on every event.
    assert data["appName"] == "Olive"
    assert data["LibraryVersion"]
    assert data["AppSessionGuid"]
    assert "appVersion" not in data
    assert "appSessionGuid" not in data


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
    assert data["deviceId"] == "DEVICE"
    assert data["deviceIdStatus"] == "ok"
    assert data["os"] == "Windows"
    assert data["osVersion"] == "10.0.22631"
    assert "leak" not in data


def test_build_payload_drops_non_scalar_whitelisted_values(tenv):
    telemetry = Telemetry()
    _quiesce(telemetry)

    payload = telemetry._build_payload(
        RECIPE_EVENT_NAME,
        {
            "recipe_name": "optimize",
            "success": True,
            "config_overrides": {
                "url": "https://user:secret@example.test/model",
                Path("/home/alice/private"): [Path("/home/alice/model.onnx")],
            },
        },
    )
    data = json.loads(payload)["data"]

    assert "configOverrides" not in data


def test_build_payload_drops_object_before_stringification(tenv):
    class SensitiveObject:
        def __str__(self):
            return r"C:\Users\Alice\private.txt"

    telemetry = Telemetry()
    _quiesce(telemetry)
    payload = telemetry._build_payload(
        RECIPE_EVENT_NAME,
        {"recipe_name": "optimize", "success": True, "model_task": SensitiveObject()},
    )

    serialized = payload.decode("utf-8")
    assert "modelTask" not in json.loads(payload)["data"]
    assert "Alice" not in serialized


def test_final_scrubber_redacts_nested_credential_aliases():
    scrubbed = scrub_value_for_telemetry(
        {
            "modelTask": {
                "access_token": "snake-secret",
                "accesstoken": "compact-secret",
                "apiKey": "camel-secret",
                "auth_header": "header-secret",
                "credential_value": "credential-secret",
                "docker_env": {"TOKEN": "env-secret"},
                "modelpath": "private/model.onnx",
                "environmentVariables": {"HOME": "/home/private"},
            }
        }
    )

    assert scrubbed["modelTask"] == {
        "access_token": "<secret>",
        "accesstoken": "<secret>",
        "apiKey": "<secret>",
        "auth_header": "<secret>",
        "credential_value": "<secret>",
        "docker_env": "<secret>",
        "environmentVariables": "<secret>",
        "modelpath": "<resource>",
    }


def test_recipe_snapshot_json_remains_parseable_after_url_redaction(tenv):
    telemetry = Telemetry()
    _quiesce(telemetry)
    snapshot = json.dumps({"auth": "token=[secret]", "endpoint": "[path]", "z_value": "kept"})

    payload = telemetry._build_payload(
        RECIPE_EVENT_NAME,
        {"recipe_name": "optimize", "success": True, "config_overrides": snapshot},
    )
    serialized_snapshot = json.loads(payload)["data"]["configOverrides"]

    assert json.loads(serialized_snapshot) == {
        "auth": "<secret>",
        "endpoint": "[path]",
        "z_value": "kept",
    }

    raw_snapshot = json.dumps(
        {
            "access_token": "snake-secret",
            "accesstoken": "compact-secret",
            "modelpath": "private/model.onnx",
        }
    )
    payload = telemetry._build_payload(
        RECIPE_EVENT_NAME,
        {"recipe_name": "optimize", "success": True, "config_overrides": raw_snapshot},
    )
    serialized_snapshot = json.loads(payload)["data"]["configOverrides"]
    assert json.loads(serialized_snapshot) == {
        "access_token": "<secret>",
        "accesstoken": "<secret>",
        "modelpath": "<resource>",
    }


def test_final_scrubber_scrubs_text_bytes_and_drops_binary():
    scrubbed = scrub_value_for_telemetry(
        {
            "text": rb"C:\Users\Alice Smith\model.onnx",
            "binary": b"\xff\x00",
        }
    )

    assert scrubbed["text"] == "[path]"
    assert scrubbed["binary"] == "[binary]"


def test_non_finite_event_is_rejected_without_affecting_next_event(tenv):
    telemetry = Telemetry()
    _quiesce(telemetry)
    telemetry._uploader = None

    telemetry.log(
        ACTION_EVENT_NAME,
        {"invoked_from": "cli", "action_name": "bad", "duration_ms": float("nan"), "success": True},
    )
    telemetry.log(
        ACTION_EVENT_NAME,
        {"invoked_from": "cli", "action_name": "good", "duration_ms": 1.0, "success": True},
    )

    assert telemetry._store.count() == 1


def test_device_id_is_canonical_shared_hash():
    import olive.telemetry.deviceid.deviceid as deviceid

    raw_id = "123e4567-e89b-42d3-a456-426614174000"
    with patch.dict(
        deviceid._device_id_state,
        {"device_id": raw_id, "status": deviceid.DeviceIdStatus.EXISTING},
        clear=True,
    ):
        hashed, status = deviceid.get_hashed_device_id_and_status()

    expected = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()
    assert hashed == f"c:{expected}"
    assert len(hashed) == 66
    assert status == deviceid.DeviceIdStatus.EXISTING


def test_device_id_hash_matches_native_algorithm_vector():
    import olive.telemetry.deviceid.deviceid as deviceid

    raw_id = "01234567-89ab-4def-8123-456789abcdef"
    with patch.dict(
        deviceid._device_id_state,
        {"device_id": raw_id, "status": deviceid.DeviceIdStatus.EXISTING},
        clear=True,
    ):
        hashed, _ = deviceid.get_hashed_device_id_and_status()

    assert hashed == f"c:{hashlib.sha256(raw_id.encode('utf-8')).hexdigest()}"
    assert hashed == "c:6225bd190d6ccf87766a49c9986d174def3391fe175a61525e49a1d2334d6a43"


def test_global_metadata_is_merged_then_filtered(tenv):
    t = Telemetry()
    _quiesce(t)

    t.add_global_metadata({"app_version": "9.9.9", "app_instance_id": "mallory@example.test", "not_allowed": "DROP"})
    payload = t._build_payload(
        ACTION_EVENT_NAME,
        {"invoked_from": "cli", "action_name": "x", "duration_ms": 1.0, "success": True},
    )
    data = json.loads(payload)["data"]
    assert data["LibraryVersion"] == tmod.VERSION
    assert data["AppSessionGuid"] == t._app_session_guid
    assert "mallory" not in payload.decode("utf-8")
    assert "not_allowed" not in data


def test_event_attributes_override_metadata(tenv):
    t = Telemetry()
    _quiesce(t)
    payload = t._build_payload(
        ERROR_EVENT_NAME,
        {"exception_type": "ValueError", "exception_message": "safe"},
        {"exception_message": r"C:\Users\Mallory\secret.txt"},
    )
    assert json.loads(payload)["data"]["exceptionMessage"] == "safe"


def test_error_event_whitelist(tenv):
    t = Telemetry()
    _quiesce(t)
    payload = t._build_payload(
        ERROR_EVENT_NAME,
        {"exception_type": "RuntimeError", "exception_message": "boom", "stack": "SENSITIVE"},
    )
    data = json.loads(payload)["data"]
    assert data["exceptionType"] == "RuntimeError"
    assert data["exceptionMessage"] == "boom"
    assert "stack" not in data


@pytest.mark.parametrize("event_name", sorted(tmod.ALLOWED_KEYS))
def test_all_whitelisted_fields_use_canonical_names(tenv, event_name):
    t = Telemetry()
    _quiesce(t)
    source = dict.fromkeys(tmod.ALLOWED_KEYS[event_name], "value")
    for snapshot_key in ("config_overrides", "package_config_overrides"):
        if snapshot_key in source:
            source[snapshot_key] = '{"key":"value"}'

    payload = t._build_payload(event_name, source)
    data = json.loads(payload)["data"]
    expected = {tmod.FIELD_NAMES.get(key, key) for key in source}
    expected.update({"appName", "LibraryVersion", "AppSessionGuid"})

    assert set(data) == expected
    for source_name, canonical_name in tmod.FIELD_NAMES.items():
        if source_name != canonical_name and source_name in source:
            assert source_name not in data


# --------------------------------------------------------------------------
# CI detection
# --------------------------------------------------------------------------


def test_is_ci_environment(monkeypatch):
    for var in (_FULL_DISABLE_VAR, _OPT_OUT_VAR, *_CI_VARS):
        monkeypatch.delenv(var, raising=False)
    assert is_ci_environment() is False
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    assert is_ci_environment() is True


@pytest.mark.parametrize("value", ["", "0", "false", " no ", "OFF"])
def test_false_ci_values_are_not_ci(monkeypatch, value):
    with patch.dict(os.environ, {"CI": value}, clear=True):
        assert is_ci_environment() is False


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


def test_store_sets_schema_version():
    store = _new_store()

    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 3


def test_store_migrates_availability_column(tmp_path):
    db_path = tmp_path / "v1.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE events (id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL)")
    conn.execute("PRAGMA user_version=1")
    conn.commit()
    conn.close()

    store = OfflineEventStore(str(db_path))

    columns = {row[1] for row in store._conn.execute("PRAGMA table_info(events)")}
    assert columns == {"id", "payload", "available_at", "acknowledged"}
    assert store._conn.execute("PRAGMA user_version").fetchone()[0] == 3
    assert store.store(b'{"migrated":1}') is True


def test_reserved_event_is_hidden_until_release():
    store = _new_store()
    row_id = store.reserve(b'{"minimal":1}', 60)

    assert row_id is not None
    assert store.count() == 1
    assert store.get_batch(10) == []

    assert store.release(row_id, b'{"enriched":1}') is True
    assert store.get_batch(10) == [(row_id, b'{"enriched":1}')]


def test_failed_delete_is_reported_and_rolled_back():
    store = _new_store()
    store.store(b'{"a":1}')
    row_id = store.get_batch(1)[0][0]
    store._conn.execute("CREATE TRIGGER fail_delete BEFORE DELETE ON events BEGIN SELECT RAISE(FAIL, 'blocked'); END")
    store._conn.commit()

    assert store.delete([row_id]) is False
    assert store.count() == 1


def test_store_operations_bound_busy_timeout_to_deadline():
    store = _new_store()
    store.store(b'{"a":1}')
    row_id = store.get_batch(1)[0][0]
    statements = []
    store._conn.set_trace_callback(statements.append)

    with patch("olive.telemetry.offline_store.time.monotonic", return_value=100.0):
        assert store.delete([row_id], deadline=100.025)

    bounded = [statement for statement in statements if statement.startswith("PRAGMA busy_timeout=")]
    assert bounded[0] in {"PRAGMA busy_timeout=24", "PRAGMA busy_timeout=25"}
    assert bounded[-1] == "PRAGMA busy_timeout=3000"


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
    result = uploader.drain_once()
    assert (result.handled, result.left, result.outcome) == (1, 0, DrainOutcome.PROGRESS)
    assert store.count() == 0


def test_uploader_retention_latch_preserves_inflight_success():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    entered = threading.Event()
    release = threading.Event()
    result = []

    def blocked_send(*_args, **_kwargs):
        _kwargs["on_send_admitted"]()
        entered.set()
        assert release.wait(5)
        return True, 204

    uploader._transport.send = blocked_send
    drain_thread = threading.Thread(target=lambda: result.append(uploader.drain_once()))
    drain_thread.start()
    assert entered.wait(5)

    uploader.retain_queued_rows()
    release.set()
    drain_thread.join(5)

    assert not drain_thread.is_alive()
    assert result[0].outcome is DrainOutcome.TRANSPORT_RETRY
    assert store.count() == 1


def test_uploader_retention_latch_prevents_new_send():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock()

    uploader.retain_queued_rows()

    assert uploader.drain_once().outcome is DrainOutcome.TRANSPORT_RETRY
    uploader._transport.send.assert_not_called()
    assert store.count() == 1


def test_uploader_retries_failed_delete_without_reposting():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock(return_value=(True, 204))
    original_delete = store.delete
    delete_attempts = 0

    def flaky_delete(ids, deadline=None):
        nonlocal delete_attempts
        delete_attempts += 1
        return False if delete_attempts == 1 else original_delete(ids, deadline)

    store.delete = flaky_delete

    assert uploader.drain_once().outcome is DrainOutcome.DELETE_RETRY
    assert uploader.drain_once().outcome is DrainOutcome.PROGRESS
    assert store.count() == 0
    uploader._transport.send.assert_called_once()


def test_uploader_retries_failed_acknowledgement_without_reposting():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock(return_value=(True, 204))
    original_acknowledge = store.acknowledge
    acknowledge_attempts = 0

    def flaky_acknowledge(ids, deadline=None):
        nonlocal acknowledge_attempts
        acknowledge_attempts += 1
        return False if acknowledge_attempts == 1 else original_acknowledge(ids, deadline)

    store.acknowledge = flaky_acknowledge

    assert uploader.drain_once().outcome is DrainOutcome.ACKNOWLEDGE_RETRY
    assert uploader.drain_once().outcome is DrainOutcome.PROGRESS
    assert store.count() == 0
    uploader._transport.send.assert_called_once()


def test_acknowledged_rows_are_deleted_without_reposting_after_restart():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock(return_value=(True, 204))
    store.delete = MagicMock(return_value=False)

    assert uploader.drain_once().outcome is DrainOutcome.DELETE_RETRY
    assert store._conn.execute("SELECT acknowledged FROM events").fetchone()[0] == 1
    db_path = store.db_path
    store.close()

    reopened = OfflineEventStore(db_path)
    restarted = EventUploader(reopened, instrumentation_key="abc-def")
    restarted._transport.send = MagicMock()
    assert restarted.drain_once().outcome is DrainOutcome.PROGRESS
    assert reopened.count() == 0
    restarted._transport.send.assert_not_called()


def test_uploader_reports_storage_read_failure():
    store, uploader = _store_and_uploader()
    store.get_batch_for_upload = MagicMock(return_value=None)

    assert uploader.drain_once().outcome is DrainOutcome.STORAGE_RETRY


def test_uploader_uses_only_remaining_deadline():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock(return_value=(False, None))

    with patch("olive.telemetry.uploader.time.monotonic", return_value=100.75):
        result = uploader.drain_once(deadline=101.0)

    assert (result.handled, result.left, result.outcome) == (0, 1, DrainOutcome.TRANSPORT_RETRY)
    assert uploader._transport.send.call_args.args[1] == pytest.approx(0.25)


def test_request_drain_only_wakes_lock_holder():
    _, uploader = _store_and_uploader()
    uploader._wake = MagicMock()
    uploader._drain_lock = MagicMock(held=False)

    uploader.request_drain()
    uploader._wake.set.assert_not_called()

    uploader._drain_lock.held = True
    uploader.request_drain()
    uploader._wake.set.assert_called_once()


def test_uploader_drops_poison_4xx():
    store, uploader = _store_and_uploader()
    store.store(b'{"bad":1}')
    uploader._transport.send = lambda *a, **k: (False, 400)
    uploader.drain_once()
    assert store.count() == 0  # dropped, not retried forever


def test_uploader_handles_oversized_single_row_without_transport():
    store, uploader = _store_and_uploader()
    store.store(b'{"oversized":true}')
    uploader._transport.send = MagicMock()

    with patch.object(OneCollectorTransportOptions, "DEFAULT_MAX_PAYLOAD_SIZE_BYTES", 1):
        result = uploader.drain_once()

    assert (result.handled, result.left, result.outcome) == (1, 0, DrainOutcome.PROGRESS)
    assert not hasattr(result, "delivered")
    assert store.count() == 0
    uploader._transport.send.assert_not_called()


def test_uploader_retains_transient_5xx():
    store, uploader = _store_and_uploader()
    store.store(b'{"later":1}')
    uploader._transport.send = lambda *a, **k: (False, 503)
    result = uploader.drain_once()
    assert (result.handled, result.left, result.outcome) == (0, 1, DrainOutcome.TRANSPORT_RETRY)
    assert store.count() == 1  # kept for retry


@pytest.mark.parametrize("status", [507, 520, 599])
def test_uploader_retains_all_server_errors(status):
    store, uploader = _store_and_uploader()
    store.store(b'{"later":1}')
    uploader._transport.send = lambda *a, **k: (False, status)

    assert uploader.drain_once().outcome is DrainOutcome.TRANSPORT_RETRY
    assert store.count() == 1


@pytest.mark.parametrize("status", [400, 413, 422])
def test_uploader_isolates_rejected_event(status):
    store, uploader = _store_and_uploader()
    store.store(b'{"bad":1}')
    store.store(b'{"valid":1}')

    def send(payload, timeout, item_count=1, on_send_admitted=None):
        if on_send_admitted is not None:
            on_send_admitted()
        if item_count > 1 or b'"bad"' in payload:
            return (False, status)
        return (True, 204)

    uploader._transport.send = MagicMock(side_effect=send)

    assert uploader.drain_once().outcome is DrainOutcome.SPLIT
    assert uploader.drain_once().outcome is DrainOutcome.PROGRESS
    assert uploader.drain_once().outcome is DrainOutcome.PROGRESS
    assert store.count() == 0
    assert uploader._transport.send.call_count == 3


@pytest.mark.parametrize("status", [400, 413, 422])
def test_flush_completes_poison_isolation_within_one_process(status):
    store, uploader = _store_and_uploader()
    store.store(b'{"bad":1}')
    store.store(b'{"valid":1}')

    def send(payload, timeout, item_count=1, on_send_admitted=None):
        if on_send_admitted is not None:
            on_send_admitted()
        if item_count > 1 or b'"bad"' in payload:
            return (False, status)
        return (True, 204)

    uploader._transport.send = MagicMock(side_effect=send)
    uploader.flush(1)

    assert store.count() == 0
    assert uploader._transport.send.call_count == 3


def test_flush_retries_acknowledged_delete_without_reposting():
    store, uploader = _store_and_uploader()
    store.store(b'{"ok":1}')
    uploader._transport.send = MagicMock(return_value=(True, 204))
    original_delete = store.delete
    delete_attempts = 0

    def flaky_delete(ids, deadline=None):
        nonlocal delete_attempts
        delete_attempts += 1
        return False if delete_attempts == 1 else original_delete(ids, deadline)

    store.delete = flaky_delete
    uploader.flush(1)

    assert store.count() == 0
    uploader._transport.send.assert_called_once()


def test_flush_does_not_touch_lock_while_thread_is_alive():
    _, uploader = _store_and_uploader()
    uploader._thread = MagicMock()
    uploader._thread.is_alive.return_value = True
    uploader._drain_lock.acquire = MagicMock()
    uploader._drain_lock.release = MagicMock()

    uploader.flush(0.01)

    uploader._drain_lock.acquire.assert_not_called()
    uploader._drain_lock.release.assert_not_called()


def test_uploader_backs_off_after_lock_contention():
    _, uploader = _store_and_uploader()
    uploader._drain_lock = MagicMock()
    uploader._drain_lock.acquire.return_value = False
    uploader._stop = MagicMock()
    uploader._stop.is_set.side_effect = [False, True]
    uploader._wake = MagicMock()

    uploader._run()

    uploader._wake.wait.assert_called_once_with(uploader._idle_backoff)


def test_uploader_backs_off_after_drain_exception():
    _, uploader = _store_and_uploader()
    uploader._drain_lock = MagicMock()
    uploader._drain_lock.acquire.return_value = True
    uploader._stop = MagicMock()
    uploader._stop.is_set.side_effect = [False, False, True]
    uploader._wake = MagicMock()
    uploader.drain_once = MagicMock(side_effect=RuntimeError("transient failure"))

    uploader._run()

    uploader._wake.wait.assert_called_once_with(uploader._idle_backoff)
    uploader._drain_lock.release.assert_called_once()


def test_uploader_backs_off_after_storage_failure():
    _, uploader = _store_and_uploader()
    uploader._drain_lock = MagicMock()
    uploader._drain_lock.acquire.return_value = True
    uploader._stop = MagicMock()
    uploader._stop.is_set.side_effect = [False, False, True]
    uploader._wake = MagicMock()
    uploader.drain_once = MagicMock(return_value=DrainResult(0, 0, DrainOutcome.STORAGE_RETRY))

    uploader._run()

    uploader._wake.wait.assert_called_once_with(uploader._idle_backoff)
    uploader._drain_lock.release.assert_called_once()


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


def test_redacted_mapping_key_collisions_are_dropped_deterministically():
    first = {"/first": "a", "/second": "b", "safe": "kept"}
    second = dict(reversed(list(first.items())))

    assert scrub_value_for_telemetry(first) == {"safe": "kept"}
    assert scrub_value_for_telemetry(second) == {"safe": "kept"}
    assert Serializer.serialize_value(first) == {"safe": "kept"}
    assert Serializer.serialize_value(second) == {"safe": "kept"}


def test_snapshot_key_collisions_are_dropped_deterministically():
    from olive.telemetry.telemetry_redaction import scrub_config_snapshot_for_telemetry

    first = {"/first": "a", "/second": "b", "safe": "kept"}
    second = dict(reversed(list(first.items())))

    assert scrub_config_snapshot_for_telemetry(first) == {"safe": "kept"}
    assert scrub_config_snapshot_for_telemetry(second) == {"safe": "kept"}


def test_create_event_envelope():
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


def test_http_retry_shares_one_timeout_budget():
    import urllib.error
    import urllib.request

    request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")
    transport = transport_mod.HttpJsonPostTransport(
        endpoint="https://example.invalid",
        ikey="key",
        compression=transport_mod.CompressionType.NO_COMPRESSION,
    )
    with (
        patch(
            "olive.telemetry.library.transport.urllib.request.urlopen",
            side_effect=urllib.error.URLError("offline"),
        ) as urlopen,
        patch("olive.telemetry.library.transport.time.monotonic", side_effect=[10.0, 10.0, 10.75]),
    ):
        assert transport._do_request(request, 1.0) == (False, None)

    assert urlopen.call_count == 2
    assert urlopen.call_args_list[0].kwargs["timeout"] == pytest.approx(1.0)
    assert urlopen.call_args_list[1].kwargs["timeout"] == pytest.approx(0.25)


def test_http_error_body_is_not_read_during_bounded_send():
    import urllib.error
    import urllib.request

    response = MagicMock()
    http_error = urllib.error.HTTPError("https://example.invalid", 503, "unavailable", {}, response)
    request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")
    transport = transport_mod.HttpJsonPostTransport(
        endpoint="https://example.invalid",
        ikey="key",
        compression=transport_mod.CompressionType.NO_COMPRESSION,
    )

    with patch("olive.telemetry.library.transport.urllib.request.urlopen", side_effect=http_error):
        assert transport._do_request(request, 1.0) == (False, 503)

    response.read.assert_not_called()
    response.close.assert_called_once()


def test_http_deadline_bounds_blocked_resolution():
    import urllib.request

    entered = threading.Event()
    release = threading.Event()
    response = MagicMock(status=204)
    response.__enter__.return_value = response
    request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")
    transport = transport_mod.HttpJsonPostTransport(
        endpoint="https://example.invalid",
        ikey="key",
        compression=transport_mod.CompressionType.NO_COMPRESSION,
    )

    def blocked_urlopen(*_args, **_kwargs):
        entered.set()
        release.wait(5)
        return response

    start = time.perf_counter()
    with patch("olive.telemetry.library.transport.urllib.request.urlopen", side_effect=blocked_urlopen):
        assert transport._do_request(request, 0.05) == (False, None)
        assert entered.wait(1)
    elapsed = time.perf_counter() - start
    release.set()

    assert elapsed < 0.5


def test_http_timeout_does_not_start_replacement_worker():
    import urllib.request

    entered = threading.Event()
    release = threading.Event()
    response = MagicMock(status=204)
    response.__enter__.return_value = response
    request = urllib.request.Request("https://example.invalid", data=b"{}", method="POST")
    transport = transport_mod.HttpJsonPostTransport(
        endpoint="https://example.invalid",
        ikey="key",
        compression=transport_mod.CompressionType.NO_COMPRESSION,
    )

    def blocked_urlopen(*_args, **_kwargs):
        entered.set()
        release.wait(5)
        return response

    with patch(
        "olive.telemetry.library.transport.urllib.request.urlopen",
        side_effect=blocked_urlopen,
    ) as urlopen:
        assert transport._do_request(request, 0.05) == (False, None)
        assert entered.wait(1)
        assert transport._do_request(request, 0.05) == (False, None)
        assert urlopen.call_count == 1
        release.set()
        transport._inflight_worker.join(1)
        assert transport._do_request(request, 0.05) == (True, 204)
        assert urlopen.call_count == 1


@pytest.mark.parametrize("status", [500, 507, 520, 599])
def test_all_server_errors_are_retryable(status):
    assert transport_mod.HttpJsonPostTransport.is_retryable(status)


def test_serialization_orders_maps_and_sets_deterministically():
    helper = Serializer
    timestamp = datetime(2025, 1, 1, tzinfo=timezone.utc)
    first = helper.create_event_envelope("event", timestamp, "o:key", {"z": {"b", "a"}, "a": 1})
    second = helper.create_event_envelope("event", timestamp, "o:key", {"a": 1, "z": {"a", "b"}})

    assert helper.serialize_to_json_bytes(first) == helper.serialize_to_json_bytes(second)


# --------------------------------------------------------------------------
# Exception-message path redaction (privacy)
# --------------------------------------------------------------------------


def test_redact_paths_and_general_length_contract():
    from olive.telemetry.telemetry_redaction import MAX_TELEMETRY_STRING_LENGTH, scrub_string_for_telemetry

    assert scrub_string_for_telemetry(r"C:\Users\alice\model.onnx") == "[path]"
    assert scrub_string_for_telemetry(r"\secret.onnx") == "[path]"
    assert scrub_string_for_telemetry(r"failed \secret.onnx") == "failed [path]"
    assert scrub_string_for_telemetry("/var/data/run/output.log") == "[path]"
    assert scrub_string_for_telemetry("/secret.onnx") == "[path]"
    # Last segment is a directory/username (no extension) -> fully redacted.
    assert scrub_string_for_telemetry("/home/bob") == "[path]"
    # UNC paths are redacted too.
    assert scrub_string_for_telemetry(r"\\server\share\secret") == "[path]"
    assert scrub_string_for_telemetry(r"failed C:\Users\Alice Smith\models\phi.onnx") == "failed [path]"
    assert scrub_string_for_telemetry("failed /home/Alice Smith/models/phi.onnx") == "failed [path]"
    assert scrub_string_for_telemetry("a/b/c") == "[path]"
    assert scrub_string_for_telemetry(r"Load Users\bob\model.onnx failed") == "Load [path]"
    assert scrub_string_for_telemetry("Users/Alice Smith/models/phi.onnx") == "[path]"
    assert scrub_string_for_telemetry("Users/Alice Smith/model.onnx") == "[path]"
    assert scrub_string_for_telemetry("models/foo.onnx") == "models/foo.onnx"
    assert scrub_string_for_telemetry("ratio 3/4 and and/or") == "ratio 3/4 and and/or"
    assert scrub_string_for_telemetry("before /home/alice/model.onnx\nafter") == "before [path]"
    assert scrub_string_for_telemetry("https://example.test/model?token=secret") == "[path]"
    assert scrub_string_for_telemetry("download(s3://private-bucket/model)") == "download([path]"
    assert scrub_string_for_telemetry("GET https://host/x?path=/home/alice/model.onnx failed") == "GET [path]"
    assert scrub_string_for_telemetry("open ./private/model.onnx") == "open [path]"
    assert scrub_string_for_telemetry(r"open ..\private\model.onnx") == "open [path]"
    assert scrub_string_for_telemetry("request token=supersecret") == "request token=[secret]"
    assert scrub_string_for_telemetry("token=first&api_key=second") == "token=[secret]"
    assert scrub_string_for_telemetry("fetch example.test?access_token=top-secret") == (
        "fetch example.test?access_token=[secret]"
    )
    assert scrub_string_for_telemetry("redirect#access_token=top-secret") == ("redirect#access_token=[secret]")
    assert scrub_string_for_telemetry("auth.token=top-secret") == "auth.token=[secret]"
    assert scrub_string_for_telemetry("auth=top-secret") == "auth=[secret]"
    assert scrub_string_for_telemetry("refreshToken=top-secret") == "refreshToken=[secret]"
    assert scrub_string_for_telemetry("AWS_SECRET_ACCESS_KEY=top-secret") == ("AWS_SECRET_ACCESS_KEY=[secret]")
    assert scrub_string_for_telemetry("PWD=odbc-value") == "PWD=[secret]"
    assert scrub_string_for_telemetry("Authorization: ******") == "Authorization: [secret]"
    assert scrub_string_for_telemetry("failure --api-key top-secret") == "failure --api-key [secret]"
    assert scrub_string_for_telemetry("Command ['tool', '--api-key', 'top-secret']") == (
        "Command ['tool', '--api-key', '[secret]"
    )
    assert scrub_string_for_telemetry('--password "-abc"') == '--password "[secret]'
    assert scrub_string_for_telemetry("Command ['tool', '--api-key', '-abc']") == (
        "Command ['tool', '--api-key', '[secret]"
    )
    assert scrub_string_for_telemetry("--password -abc") == "--password -abc"
    assert scrub_string_for_telemetry("model dir /_apikey:top-secret") == "model dir [path]"
    assert scrub_string_for_telemetry("arg /1token=top-secret") == "arg [path]"
    assert scrub_string_for_telemetry("connect /2fa_token=top-secret") == "connect [path]"
    assert scrub_string_for_telemetry("failure /password:top-secret") == "failure /password:[secret]"
    assert scrub_string_for_telemetry("connect user:password@localhost/model") == "connect [secret]"
    assert scrub_string_for_telemetry("n/a read/write domain\\user") == "n/a read/write domain\\user"
    assert scrub_string_for_telemetry("meta-llama/Llama-3.1-8B-Instruct") == "meta-llama/Llama-3.1-8B-Instruct"
    assert scrub_string_for_telemetry("tokenizer=enabled oauth=enabled") == "tokenizer=enabled oauth=enabled"
    assert (
        len(scrub_string_for_telemetry("x" * (MAX_TELEMETRY_STRING_LENGTH + 100)).encode("utf-8"))
        == MAX_TELEMETRY_STRING_LENGTH
    )
    assert scrub_string_for_telemetry("x" * (MAX_TELEMETRY_STRING_LENGTH - 1) + "€") == (
        "x" * (MAX_TELEMETRY_STRING_LENGTH - 1)
    )
    redacted_at_limit = scrub_string_for_telemetry("x" * (MAX_TELEMETRY_STRING_LENGTH - 5) + " /a/b/c")
    assert len(redacted_at_limit.encode("utf-8")) == MAX_TELEMETRY_STRING_LENGTH
    assert redacted_at_limit.endswith("[path]")


def test_secret_scanner_advances_past_rejected_key_tokens():
    from olive.telemetry import telemetry_redaction

    value = "-" + "field-" * 6_800 + "value"
    with patch(
        "olive.telemetry.telemetry_redaction.is_sensitive_config_key_for_telemetry",
        wraps=telemetry_redaction.is_sensitive_config_key_for_telemetry,
    ) as is_sensitive_key:
        scrubbed = telemetry_redaction.scrub_string_for_telemetry(value)

    assert scrubbed == value
    assert is_sensitive_key.call_count == 1


def test_error_messages_are_capped_at_40960_utf8_bytes():
    from olive.telemetry.telemetry_extensions import log_error
    from olive.telemetry.telemetry_redaction import MAX_ERROR_MESSAGE_LENGTH

    telemetry = MagicMock()
    with patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry):
        log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH + 100))
        truncated = telemetry.log.call_args.args[1]["exception_message"]
        assert len(truncated.encode("utf-8")) == MAX_ERROR_MESSAGE_LENGTH

        log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH - 1) + "€")
        multibyte = telemetry.log.call_args.args[1]["exception_message"]
        assert multibyte == "x" * (MAX_ERROR_MESSAGE_LENGTH - 1)

        log_error("RuntimeError", "x" * (MAX_ERROR_MESSAGE_LENGTH - 5) + " /a/b/c")
        redacted_at_limit = telemetry.log.call_args.args[1]["exception_message"]
        assert len(redacted_at_limit.encode("utf-8")) == MAX_ERROR_MESSAGE_LENGTH
        assert redacted_at_limit.endswith("[path]")


def test_error_payload_preserves_error_specific_size_limit(tenv):
    from olive.telemetry.telemetry_redaction import MAX_ERROR_MESSAGE_LENGTH

    telemetry = Telemetry()
    _quiesce(telemetry)
    payload = telemetry._build_payload(
        ERROR_EVENT_NAME,
        {
            "exception_type": "RuntimeError",
            "exception_message": "x" * (MAX_ERROR_MESSAGE_LENGTH + 100),
        },
    )

    assert len(json.loads(payload)["data"]["exceptionMessage"].encode("utf-8")) == MAX_ERROR_MESSAGE_LENGTH


def test_format_exception_message_redacts_paths_in_message():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    exc = RuntimeError(r"failed to read C:\Users\alice\secret\weights.bin")
    message = _format_exception_message(exc, exc.__traceback__)
    assert "alice" not in message
    assert "[path]" in message


def test_format_exception_message_handles_unprintable_exception():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    class UnprintableError(Exception):
        def __str__(self):
            raise RuntimeError("cannot render")

    assert _format_exception_message(UnprintableError()).endswith("UnprintableError: <exception str() failed>")


def test_public_helpers_never_propagate_failures():
    from olive.telemetry.telemetry_extensions import log_action, log_error, log_recipe_result

    with patch("olive.telemetry.telemetry_extensions._get_logger", side_effect=RuntimeError("telemetry failed")):
        log_action("test", "work", 1.0, True, metadata=["not", "a", "dict"])
        log_error("RuntimeError", "boom", metadata=["not", "a", "dict"])
        log_recipe_result("recipe", True, metadata=["not", "a", "dict"])


def test_log_recipe_result_never_propagates_when_telemetry_log_raises():
    from olive.telemetry.telemetry_extensions import log_recipe_result

    telemetry = MagicMock()
    telemetry.log.side_effect = RuntimeError("log failed")
    with patch("olive.telemetry.telemetry_extensions._get_logger", return_value=telemetry):
        log_recipe_result("recipe", True)

    telemetry.log.assert_called_once()


def _raise_error_called_with_source_secret(_secret):
    raise RuntimeError("boom")


def test_format_exception_message_omits_source_code():
    from olive.telemetry.telemetry_extensions import _format_exception_message

    try:
        _raise_error_called_with_source_secret("source-secret")
    except RuntimeError as ex:
        message = _format_exception_message(ex, ex.__traceback__)

    assert "source-secret" not in message
    assert __file__ not in message
    assert 'File "[path]"' in message
    assert "in _raise_error_called_with_source_secret" in message
    assert message.endswith("RuntimeError: boom")


def test_device_id_store_uses_owner_only_creation_mode(tmp_path):
    with (
        patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path),
        patch.object(Path, "mkdir") as mock_mkdir,
    ):
        deviceid_store_mod.Store().store_id("test-device-id")

    mock_mkdir.assert_called_once_with(mode=0o700, parents=True, exist_ok=True)


def test_relative_cache_environment_uses_absolute_home(tmp_path, monkeypatch):
    telemetry_utils.get_telemetry_base_dir.cache_clear()
    monkeypatch.setenv("XDG_CACHE_HOME", "relative-cache")
    monkeypatch.setenv("HOME", str(tmp_path))
    with patch.object(telemetry_utils.platform, "system", return_value="Linux"):
        base_dir = telemetry_utils.get_telemetry_base_dir()
    telemetry_utils.get_telemetry_base_dir.cache_clear()

    assert base_dir == tmp_path / ".cache" / telemetry_utils.ORT_SUPPORT_DIR
    assert base_dir.is_absolute()


def test_relative_windows_appdata_uses_absolute_home(tmp_path, monkeypatch):
    telemetry_utils.get_telemetry_base_dir.cache_clear()
    monkeypatch.setenv("LOCALAPPDATA", "relative-local")
    monkeypatch.setenv("APPDATA", "relative-roaming")
    monkeypatch.setenv("HOME", "relative-home")
    with (
        patch.object(telemetry_utils.platform, "system", return_value="Windows"),
        patch.object(telemetry_utils.Path, "home", return_value=tmp_path),
    ):
        base_dir = telemetry_utils.get_telemetry_base_dir()
    telemetry_utils.get_telemetry_base_dir.cache_clear()

    assert base_dir == tmp_path / "AppData" / "Local" / telemetry_utils.ORT_SUPPORT_DIR
    assert base_dir.is_absolute()


def test_windows_telemetry_base_dir_uses_canonical_developer_tools_path(tmp_path, monkeypatch):
    telemetry_utils.get_telemetry_base_dir.cache_clear()
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    with patch.object(telemetry_utils.platform, "system", return_value="Windows"):
        base_dir = telemetry_utils.get_telemetry_base_dir()
    telemetry_utils.get_telemetry_base_dir.cache_clear()

    assert base_dir == tmp_path / telemetry_utils.ORT_SUPPORT_DIR


def test_home_resolution_fails_without_absolute_per_user_directory(monkeypatch):
    fake_pwd = SimpleNamespace(getpwuid=MagicMock(side_effect=KeyError("missing")))
    monkeypatch.setenv("HOME", "relative-home")
    with (
        patch.object(telemetry_utils.platform, "system", return_value="Linux"),
        patch.object(telemetry_utils.Path, "home", return_value=Path("relative-home")),
        patch.dict(sys.modules, {"pwd": fake_pwd}),
        pytest.raises(RuntimeError, match="No absolute per-user"),
    ):
        telemetry_utils._resolve_home_dir()


def test_device_id_is_ephemeral_when_per_user_storage_is_unavailable():
    import olive.telemetry.deviceid.deviceid as deviceid

    deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})
    with (
        patch.object(deviceid.platform, "system", return_value="Linux"),
        patch.object(deviceid, "Store", side_effect=RuntimeError("no home")),
    ):
        generated = deviceid.get_device_id()

    assert deviceid._is_valid_device_id(generated)
    assert deviceid._device_id_state["status"] == deviceid.DeviceIdStatus.FAILED


def test_missing_device_id_raises_file_not_found(tmp_path):
    with (
        patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path),
        pytest.raises(FileNotFoundError),
    ):
        _ = deviceid_store_mod.Store().retrieve_id


def test_corrupted_device_id_is_atomically_repaired(tmp_path):
    import olive.telemetry.deviceid.deviceid as deviceid

    device_id_path = tmp_path / "deviceid"
    device_id_path.write_bytes(b"\xff\xfe")
    deviceid._device_id_state.update({"device_id": None, "status": deviceid.DeviceIdStatus.NEW})

    with (
        patch.object(deviceid.platform, "system", return_value="Linux"),
        patch.object(deviceid, "get_telemetry_base_dir", return_value=tmp_path),
        patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path),
    ):
        repaired = deviceid.get_device_id()

    assert deviceid._is_valid_device_id(repaired)
    assert device_id_path.read_text(encoding="utf-8") == repaired
    assert deviceid._device_id_state["status"] == deviceid.DeviceIdStatus.CORRUPTED


def test_file_store_does_not_overwrite_concurrent_winner(tmp_path):
    with patch.object(deviceid_store_mod, "get_telemetry_base_dir", return_value=tmp_path):
        store = deviceid_store_mod.Store()
        assert store.store_id("first") is True
        assert store.store_id("second") is False

    assert (tmp_path / "deviceid").read_text(encoding="utf-8") == "first"


def test_concurrent_processes_publish_one_device_id(tmp_path):
    script = (
        "import platform; "
        "import olive.telemetry.deviceid.deviceid as d; "
        "import olive.telemetry.utils as u; "
        "platform.system=lambda:'Linux'; "
        "u.get_telemetry_base_dir.cache_clear(); "
        "print(d.get_device_id())"
    )
    env = os.environ.copy()
    env["XDG_CACHE_HOME"] = str(tmp_path)

    def run_process(_):
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
            check=False,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        completed = list(executor.map(run_process, range(6)))
    results = []
    for process in completed:
        assert process.returncode == 0, process.stderr
        results.append(process.stdout.strip())

    assert len(set(results)) == 1


@pytest.mark.skipif(os.name != "nt", reason="Windows named mutex")
def test_windows_device_id_protocol_serializes_cross_process_publication(tmp_path):
    script = """
import os
from pathlib import Path
import olive.telemetry.deviceid.deviceid as d

class FileBackedWindowsStore:
    @property
    def retrieve_id(self):
        path = Path(os.environ["OLIVE_TEST_DEVICE_ID"])
        if not path.exists():
            raise FileNotFoundError
        return path.read_text(encoding="utf-8")

    def store_id(self, device_id, replace_existing=False):
        Path(os.environ["OLIVE_TEST_DEVICE_ID"]).write_text(device_id, encoding="utf-8")
        return True

d.platform.system = lambda: "Windows"
d.WindowsStore = FileBackedWindowsStore
d.get_telemetry_base_dir = lambda: Path(os.environ["OLIVE_TEST_DEVICE_DIR"])
print(d.get_device_id())
"""
    env = os.environ.copy()
    env["OLIVE_TEST_DEVICE_DIR"] = str(tmp_path)
    env["OLIVE_TEST_DEVICE_ID"] = str(tmp_path / "registry-deviceid")

    def run_process(_):
        return subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            env=env,
            timeout=90,
            check=False,
        )

    with ThreadPoolExecutor(max_workers=6) as executor:
        completed = list(executor.map(run_process, range(6)))
    results = []
    for process in completed:
        assert process.returncode == 0, process.stderr
        results.append(process.stdout.strip())

    assert len(set(results)) == 1


def test_windows_device_id_store_uses_least_privilege_access():
    winreg = MagicMock(
        HKEY_CURRENT_USER=object(),
        KEY_QUERY_VALUE=0x0001,
        KEY_SET_VALUE=0x0002,
        KEY_CREATE_SUB_KEY=0x0004,
        KEY_WOW64_64KEY=0x0100,
        REG_SZ=1,
    )
    winreg.QueryValueEx.side_effect = FileNotFoundError
    key_handle = object()
    winreg.CreateKeyEx.return_value.__enter__.return_value = key_handle

    with patch.dict("sys.modules", {"winreg": winreg}):
        deviceid_store_mod.WindowsStore().store_id("test-device-id")

    winreg.CreateKeyEx.assert_called_once_with(
        winreg.HKEY_CURRENT_USER,
        deviceid_store_mod.REGISTRY_PATH,
        reserved=0,
        access=winreg.KEY_QUERY_VALUE | winreg.KEY_SET_VALUE | winreg.KEY_CREATE_SUB_KEY | winreg.KEY_WOW64_64KEY,
    )
    winreg.SetValueEx.assert_called_once_with(
        key_handle,
        deviceid_store_mod.REGISTRY_KEY,
        0,
        winreg.REG_SZ,
        "test-device-id",
    )


def test_windows_device_id_store_preserves_existing_value():
    winreg = MagicMock(
        HKEY_CURRENT_USER=object(),
        KEY_QUERY_VALUE=0x0001,
        KEY_SET_VALUE=0x0002,
        KEY_CREATE_SUB_KEY=0x0004,
        KEY_WOW64_64KEY=0x0100,
        REG_SZ=1,
    )
    winreg.QueryValueEx.return_value = ("existing-device-id", winreg.REG_SZ)

    with patch.dict("sys.modules", {"winreg": winreg}):
        deviceid_store_mod.WindowsStore().store_id("new-device-id")

    winreg.SetValueEx.assert_not_called()


def test_windows_device_id_store_replaces_existing_value_when_requested():
    winreg = MagicMock(
        HKEY_CURRENT_USER=object(),
        KEY_QUERY_VALUE=0x0001,
        KEY_SET_VALUE=0x0002,
        KEY_CREATE_SUB_KEY=0x0004,
        KEY_WOW64_64KEY=0x0100,
        REG_SZ=1,
    )
    key_handle = object()
    winreg.CreateKeyEx.return_value.__enter__.return_value = key_handle

    with patch.dict("sys.modules", {"winreg": winreg}):
        deviceid_store_mod.WindowsStore().store_id("new-device-id", replace_existing=True)

    winreg.QueryValueEx.assert_not_called()
    winreg.SetValueEx.assert_called_once_with(
        key_handle,
        deviceid_store_mod.REGISTRY_KEY,
        0,
        winreg.REG_SZ,
        "new-device-id",
    )


def test_windows_device_id_store_rejects_wrong_registry_type():
    winreg = MagicMock(
        HKEY_CURRENT_USER=object(),
        KEY_READ=0x0001,
        KEY_WOW64_64KEY=0x0100,
        REG_SZ=1,
        REG_BINARY=3,
    )
    winreg.QueryValueEx.return_value = (b"not-a-string", winreg.REG_BINARY)

    with (
        patch.dict("sys.modules", {"winreg": winreg}),
        pytest.raises(ValueError, match="not a string"),
    ):
        _ = deviceid_store_mod.WindowsStore().retrieve_id


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
