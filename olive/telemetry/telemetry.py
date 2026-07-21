# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Telemetry singleton backed by a durable SQLite event queue.

Events are serialized to Common Schema JSON and written to a per-app SQLite
store; a background uploader drains the store to Microsoft OneCollector. Because
every event is persisted before any network call, the process can exit at any
time without losing data and without an exit-time flush. The pipeline uses only
the Python standard library (no OpenTelemetry, no requests).
"""

import base64
import os
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from olive.telemetry.constants import CONNECTION_STRING
from olive.telemetry.deviceid import get_encrypted_device_id_and_status
from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.options import CompressionType, OneCollectorExporterOptions, OneCollectorTransportOptions
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper
from olive.telemetry.library.transport import HttpJsonPostTransport
from olive.telemetry.offline_store import OfflineEventStore
from olive.telemetry.uploader import EventUploader
from olive.telemetry.utils import get_telemetry_base_dir

try:
    from olive.version import __version__ as VERSION
except Exception:
    VERSION = "unknown"

# Default event names used by the high-level telemetry helpers.
HEARTBEAT_EVENT_NAME = "OliveHeartbeat"
RECIPE_EVENT_NAME = "OliveRecipe"
ACTION_EVENT_NAME = "OliveAction"
ERROR_EVENT_NAME = "OliveError"
APP_NAME = "Olive"

# CI/CD environment variables whose presence indicates an automated pipeline.
_CI_ENV_VARS = (
    "CI",  # GitHub Actions, GitLab CI, Travis CI, CircleCI, generic
    "TF_BUILD",  # Azure Pipelines
    "GITHUB_ACTIONS",  # GitHub Actions
    "JENKINS_URL",  # Jenkins
    "CODEBUILD_BUILD_ID",  # AWS CodeBuild
    "BUILDKITE",  # Buildkite
    "SYSTEM_TEAMFOUNDATIONCOLLECTIONURI",  # Azure DevOps
)

ALLOWED_KEYS = {
    HEARTBEAT_EVENT_NAME: {
        "device_id",
        "device_id_status",
        "os",
        "os_version",
        "os_release",
        "os_arch",
        "app_version",
        "app_instance_id",
        "initTs",
    },
    ACTION_EVENT_NAME: {
        "invoked_from",
        "action_name",
        "duration_ms",
        "success",
        "app_version",
        "app_instance_id",
        "initTs",
    },
    ERROR_EVENT_NAME: {
        "exception_type",
        "exception_message",
        "app_version",
        "app_instance_id",
        "initTs",
    },
    RECIPE_EVENT_NAME: {
        "recipe_name",
        "recipe_hash",
        "recipe_source",
        "recipe_format",
        "recipe_command",
        "execution_mode",
        "workflow_id",
        "config_overrides",
        "success",
        "input_model_type",
        "input_model_source",
        "model_task",
        "target_system_type",
        "target_device",
        "target_execution_provider",
        "target_execution_providers",
        "host_system_type",
        "host_device",
        "host_execution_provider",
        "host_execution_providers",
        "pass_types",
        "pass_count",
        "data_config_count",
        "search_enabled",
        "package_config_provided",
        "package_config_overrides",
        "is_ci",
        "app_version",
        "app_instance_id",
        "initTs",
    },
}

CRITICAL_EVENTS = {HEARTBEAT_EVENT_NAME}

# Per-app database file. Olive and other apps use separate files so a process
# never drains another app's events (which carry a different tenant key).
DB_FILE_NAME = "olive_telemetry.db"


def is_ci_environment() -> bool:
    """Detect CI/CD environments by checking well-known environment variables."""
    return any(os.environ.get(var) for var in _CI_ENV_VARS)


class Telemetry:
    """Per-process singleton that persists events to SQLite and uploads them.

    Separate processes get separate in-memory singletons and coordinate only
    through the shared SQLite store and its single-drainer file lock.
    Use Telemetry() to get the singleton instance.
    """

    _instance: Optional["Telemetry"] = None
    _lock = threading.RLock()

    @classmethod
    def get_existing_instance(cls) -> Optional["Telemetry"]:
        """Return the current singleton without creating telemetry."""
        return cls._instance

    def __new__(cls):
        """Create or return the singleton instance."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    instance = super().__new__(cls)
                    instance._initialized = False
                    cls._instance = instance
        return cls._instance

    def __init__(self):
        """Initialize the telemetry store and uploader (runs once)."""
        with self._lock:
            if self._initialized:
                return
            # Mark initialized under the lock before doing any work, so two
            # threads whose first Telemetry() calls interleave cannot both run
            # the body (which would create two uploaders and two heartbeats).
            self._initialized = True

            self._store: Optional[OfflineEventStore] = None
            self._uploader: Optional[EventUploader] = None
            self._enabled = True
            self._recipe_only_ci_telemetry = False
            self._global_metadata: dict[str, Any] = {}
            self._instrumentation_key = ""
            self._envelope_ikey = ""
            self._app_instance_id = uuid.uuid4().hex
            self._heartbeat_thread: Optional[threading.Thread] = None

            try:
                # User opt-out (OLIVE_DISABLE_TELEMETRY=1): detailed events are
                # not recorded, but the device-id heartbeat is still sent
                # directly so device counting keeps working without opening or
                # draining the durable detailed-event store. CI is handled via
                # recipe-only mode below and never sends a heartbeat.
                user_opt_out = os.environ.get("OLIVE_DISABLE_TELEMETRY") == "1"

                options = OneCollectorExporterOptions(connection_string=base64.b64decode(CONNECTION_STRING).decode())
                options.validate()
                self._instrumentation_key = options.instrumentation_key
                self._envelope_ikey = (
                    f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"
                )

                event_source.disable()

                # In CI, only recipe events are sent (no heartbeat, no
                # action/error); this is independent of user opt-out.
                self._recipe_only_ci_telemetry = is_ci_environment()

                # Opt-out + CI: record and send nothing at all.
                if user_opt_out and self._recipe_only_ci_telemetry:
                    self._enabled = False
                    return

                # Detailed events are recorded only when enabled; the heartbeat
                # ignores this gate.
                self._enabled = not user_opt_out

                if user_opt_out:
                    self._start_heartbeat(durable=False)
                    return

                # Durable on-disk queue + background uploader. The uploader
                # retries enabled-run events until delivery.
                db_path = os.path.join(get_telemetry_base_dir(), DB_FILE_NAME)
                self._store = OfflineEventStore(db_path)
                if not self._store.is_open:
                    self._store = None
                    self._enabled = False
                    return
                self._uploader = EventUploader(self._store, instrumentation_key=self._instrumentation_key)
                self._uploader.start()

                # The device-id heartbeat is written to the durable store, not
                # sent directly. It is suppressed in CI (recipe-only mode).
                if not self._recipe_only_ci_telemetry:
                    self._start_heartbeat(durable=True)
            except Exception:
                # Fail silently — telemetry must never crash the host application
                self._store = None
                self._uploader = None
                self._enabled = False

    def _start_heartbeat(self, durable: bool) -> None:
        """Send the device-id heartbeat on a background daemon thread."""
        self._heartbeat_thread = threading.Thread(
            target=self._send_heartbeat, args=(None, durable), name="olive-telemetry-heartbeat", daemon=True
        )
        self._heartbeat_thread.start()

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        """Merge metadata into every subsequent telemetry event."""
        try:
            if metadata:
                self._global_metadata = {**self._global_metadata, **metadata}
        except Exception:
            pass

    @property
    def accepts_detailed_events(self) -> bool:
        """Whether action and error events can currently be persisted."""
        return bool(
            self._enabled and not self._recipe_only_ci_telemetry and self._store is not None and self._store.is_open
        )

    def log(
        self,
        event_name: str,
        attributes: Optional[dict[str, Any]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a telemetry event (persisted durably, uploaded in the background)."""
        try:
            if not self._enabled or self._store is None:
                return
            if self._recipe_only_ci_telemetry and event_name != RECIPE_EVENT_NAME:
                return
            payload = self._build_payload(event_name, attributes, metadata)
            if payload is None:
                return
            self._store.store(payload)
            if self._uploader is not None:
                self._uploader.request_drain()
        except Exception:
            # Fail silently — telemetry must never crash the host application
            pass

    def _build_payload(
        self,
        event_name: str,
        attributes: Optional[dict[str, Any]],
        metadata: Optional[dict[str, Any]] = None,
    ) -> Optional[bytes]:
        """Merge metadata, filter to whitelisted keys, and serialize one event.

        Returns the Common Schema JSON bytes, or None if the event is not
        whitelisted or filters to nothing.
        """
        attrs = _merge_metadata(attributes, metadata)
        if self._global_metadata:
            attrs = {**self._global_metadata, **attrs}
        filtered = _filter_event_data(event_name, attrs)
        if not filtered:
            # Unknown/empty event: not whitelisted.
            return None
        filtered.setdefault("app_version", VERSION)
        filtered.setdefault("app_instance_id", self._app_instance_id)
        envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
            event_name=event_name,
            timestamp=datetime.now(timezone.utc),
            ikey=self._envelope_ikey,
            data=filtered,
        )
        return CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)

    def _send_heartbeat(self, metadata: Optional[dict[str, Any]] = None, durable: bool = True) -> None:
        """Send the device-id heartbeat.

        Enabled runs enqueue it in the durable store. User opt-out sends it
        directly so disabled runs never drain queued detailed events from an
        earlier enabled run.
        """
        if durable and self._store is None:
            return
        try:
            encrypted_device_id, device_id_status = get_encrypted_device_id_and_status()
            attributes = {
                "device_id": encrypted_device_id,
                "device_id_status": device_id_status.value,
                "os": platform.system(),
                "os_version": platform.version(),
                "os_release": platform.release(),
                "os_arch": platform.machine(),
            }
            payload = self._build_payload(HEARTBEAT_EVENT_NAME, attributes, metadata)
            if payload is None:
                return
            if durable:
                self._store.store(payload)
                if self._uploader is not None:
                    self._uploader.request_drain()
            else:
                transport = HttpJsonPostTransport(
                    endpoint=OneCollectorTransportOptions.DEFAULT_ENDPOINT,
                    ikey=self._instrumentation_key,
                    compression=CompressionType.DEFLATE,
                )
                transport.send(payload, timeout_sec=2.0, item_count=1)
        except Exception:
            pass

    def disable_telemetry(self) -> None:
        """Disable telemetry and stop the background uploader (non-blocking)."""
        try:
            self._enabled = False
            if self._uploader is not None:
                # Non-blocking: signal the daemon thread to wind down without
                # joining, so opting out never blocks the caller.
                self._uploader.signal_stop()
        except Exception:
            pass

    def shutdown(
        self,
        timeout_millis: float = 10_000,
        callback_timeout_millis: float = 2_000,
        flush_seconds: float = 0,
    ) -> None:
        """Stop the background uploader with bounded cleanup.

        Delivery does not depend on a flush here: durability guarantees that any
        undelivered events remain in the on-disk store and are uploaded on the
        next run (or by a concurrently-running process). Synchronous network I/O
        occurs only when a caller explicitly supplies ``flush_seconds`` (used by
        ephemeral Docker runners).
        """
        try:
            timeout_seconds = max(0.0, timeout_millis / 1000.0)
            callback_timeout_seconds = max(0.0, callback_timeout_millis / 1000.0)
            flush_seconds = max(0.0, flush_seconds)
            deadline = time.monotonic() + max(timeout_seconds, callback_timeout_seconds, flush_seconds)

            def remaining_seconds() -> float:
                return max(0.0, deadline - time.monotonic())

            heartbeat_stopped = True
            if self._heartbeat_thread is not None and self._heartbeat_thread is not threading.current_thread():
                self._heartbeat_thread.join(min(callback_timeout_seconds, remaining_seconds()))
                heartbeat_stopped = not self._heartbeat_thread.is_alive()
                if heartbeat_stopped:
                    self._heartbeat_thread = None

            uploader_stopped = True
            if self._uploader is not None:
                uploader_stopped = self._uploader.stop_loop(
                    join_timeout_seconds=min(timeout_seconds, remaining_seconds())
                )
                if uploader_stopped:
                    if flush_seconds > 0:
                        flush_timeout = min(flush_seconds, remaining_seconds())
                        if flush_timeout > 0:
                            self._uploader.flush(flush_timeout)
                    self._uploader.close()
                    self._uploader = None
            if self._store is not None and uploader_stopped and heartbeat_stopped:
                self._store.close()
                self._store = None
        except Exception:
            # Fail silently — telemetry must never crash the host application
            pass

    def __del__(self):
        """Safety-net cleanup on garbage collection."""
        try:
            self.shutdown(timeout_millis=0, callback_timeout_millis=0, flush_seconds=0)
        except Exception:
            pass


def _get_logger() -> Telemetry:
    """Get or create the singleton Telemetry instance."""
    return Telemetry()


def _merge_metadata(attributes: Optional[dict[str, Any]], metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(metadata or {})
    if attributes:
        merged.update(attributes)
    return merged


def _filter_event_data(event_name: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Filter event data to only allowed keys for privacy/security.

    Whitelist approach: only explicitly allowed keys (with dot-notation support
    for nested values, e.g. "os.name") are kept. Returns None for unknown events
    so they are neither persisted nor sent.
    """
    if event_name not in ALLOWED_KEYS:
        return None
    allowed_keys = ALLOWED_KEYS[event_name]

    filtered: dict[str, Any] = {}
    for key in allowed_keys:
        value = _get_nested_value(data, key)
        if value is None:
            continue
        _set_nested_value(filtered, key, value)
    return filtered or None


def _get_nested_value(data: dict[str, Any], key: str) -> Any:
    current = data
    for part in key.split("."):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _set_nested_value(data: dict[str, Any], key: str, value: Any) -> None:
    current = data
    parts = key.split(".")
    for part in parts[:-1]:
        current = current.setdefault(part, {})
    current[parts[-1]] = value
