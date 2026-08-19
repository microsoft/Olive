# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Telemetry singleton backed by a durable SQLite event queue.

Detailed events are serialized to Common Schema JSON and written to a per-app
SQLite store; a background uploader drains the store to Microsoft OneCollector.
The one-per-process Heartbeat uses the same durable queue. The pipeline uses only
the Python standard library (no OpenTelemetry, no requests).
"""

import base64
import json
import os
import platform
import threading
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from olive.telemetry.deviceid import get_hashed_device_id_and_status
from olive.telemetry.library.options import OneCollectorExporterOptions
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper
from olive.telemetry.offline_store import OfflineEventStore
from olive.telemetry.telemetry_redaction import (
    MAX_TELEMETRY_STRING_LENGTH,
    scrub_config_snapshot_for_telemetry,
    scrub_error_message_for_telemetry,
    scrub_value_for_telemetry,
)
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

# CI/CD environment variables whose presence indicates an automated pipeline.
_CI_ENV_VARS = (
    "CI",  # GitHub Actions, GitLab CI, Travis CI, CircleCI, generic
    "TF_BUILD",  # Azure Pipelines
    "GITHUB_ACTIONS",  # GitHub Actions
    "GITLAB_CI",  # GitLab CI
    "CIRCLECI",  # CircleCI
    "TRAVIS",  # Travis CI
    "JENKINS_URL",  # Jenkins
    "CODEBUILD_BUILD_ID",  # AWS CodeBuild
    "BUILDKITE",  # Buildkite
    "TEAMCITY_VERSION",  # TeamCity
    "APPVEYOR",  # AppVeyor
    "BITBUCKET_BUILD_NUMBER",  # Bitbucket Pipelines
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
        "initTs",
    },
    ACTION_EVENT_NAME: {
        "invoked_from",
        "action_name",
        "duration_ms",
        "success",
        "initTs",
    },
    ERROR_EVENT_NAME: {
        "exception_type",
        "exception_message",
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
        "initTs",
    },
}

FIELD_NAMES = {
    "device_id": "deviceId",
    "device_id_status": "deviceIdStatus",
    "os_version": "osVersion",
    "os_release": "osRelease",
    "os_arch": "osArchitecture",
    "invoked_from": "invokedFrom",
    "action_name": "actionName",
    "duration_ms": "durationMs",
    "exception_type": "exceptionType",
    "exception_message": "exceptionMessage",
    "recipe_name": "recipeName",
    "recipe_hash": "recipeHash",
    "recipe_source": "recipeSource",
    "recipe_format": "recipeFormat",
    "recipe_command": "recipeCommand",
    "execution_mode": "executionMode",
    "workflow_id": "workflowId",
    "config_overrides": "configOverrides",
    "input_model_type": "inputModelType",
    "input_model_source": "inputModelSource",
    "model_task": "modelTask",
    "target_system_type": "targetSystemType",
    "target_device": "targetDevice",
    "target_execution_provider": "targetExecutionProvider",
    "target_execution_providers": "targetExecutionProviders",
    "host_system_type": "hostSystemType",
    "host_device": "hostDevice",
    "host_execution_provider": "hostExecutionProvider",
    "host_execution_providers": "hostExecutionProviders",
    "pass_types": "passTypes",
    "pass_count": "passCount",
    "data_config_count": "dataConfigCount",
    "search_enabled": "searchEnabled",
    "package_config_provided": "packageConfigProvided",
    "package_config_overrides": "packageConfigOverrides",
    "is_ci": "isCI",
    "app_version": "LibraryVersion",
    "app_instance_id": "AppSessionGuid",
}

# Per-app database file. Olive and other apps use separate files so a process
# never drains another app's events (which carry a different tenant key).
DB_FILE_NAME = "olive_telemetry.db"
CI_DB_FILE_NAME = "olive_recipe_telemetry.db"
_HEARTBEAT_RELEASE_SECONDS = 60.0


def _is_environment_signal_truthy(value: str) -> bool:
    return value.strip().lower() not in {"", "0", "false", "no", "off"}


def is_ci_environment() -> bool:
    """Detect CI/CD environments by checking well-known environment variables."""
    return any(_is_environment_signal_truthy(os.environ.get(var, "")) for var in _CI_ENV_VARS)


def is_telemetry_disabled_by_environment() -> bool:
    """Return whether any supported environment variable requests full suppression."""
    return any(
        os.environ.get(variable, "").strip().lower() in {"1", "true", "yes", "on", "y"}
        for variable in ("ORT_DISABLE_TELEMETRY", "OLIVE_DISABLE_TELEMETRY")
    )


class Telemetry:
    """Per-process singleton that persists events to SQLite and uploads them.

    Separate processes get separate in-memory singletons and coordinate only
    through the shared SQLite store and its single-drainer file lock.
    Use Telemetry() to get the singleton instance.
    """

    _instance: Optional["Telemetry"] = None
    _lock = threading.RLock()
    _process_disabled = False
    _heartbeat_enqueued = False

    @classmethod
    def get_existing_instance(cls) -> Optional["Telemetry"]:
        """Return the current singleton without creating telemetry."""
        return cls._instance

    @classmethod
    def get_or_create_if_enabled(cls) -> Optional["Telemetry"]:
        """Return the singleton only when process telemetry is not fully disabled."""
        with cls._lock:
            if cls._process_disabled or is_telemetry_disabled_by_environment():
                cls._process_disabled = True
                if cls._instance is not None:
                    cls._instance.disable_telemetry()
                return None
            return cls()

    def __new__(cls):
        """Create or return the singleton instance."""
        with cls._lock:
            if cls._process_disabled or is_telemetry_disabled_by_environment():
                cls._process_disabled = True
                if cls._instance is not None:
                    cls._instance.disable_telemetry()
                    return cls._instance
                return cls._new_unpublished_instance(disabled=True, initialized=True)
            if cls._instance is None:
                cls._instance = cls._new_unpublished_instance(disabled=False, initialized=False)
            return cls._instance

    @classmethod
    def _new_unpublished_instance(cls, *, disabled: bool, initialized: bool) -> "Telemetry":
        instance = super().__new__(cls)
        instance._initialized = initialized
        instance._disabled = disabled
        instance._store = None
        instance._uploader = None
        instance._enabled = not disabled
        instance._recipe_only_ci_telemetry = False
        instance._global_metadata = {}
        instance._instrumentation_key = ""
        instance._envelope_ikey = ""
        instance._app_session_guid = ""
        return instance

    def __init__(self):
        """Initialize the telemetry store and uploader (runs once)."""
        with self._lock:
            if self._initialized:
                return
            # Mark initialized under the lock before doing any work, so two
            # threads whose first Telemetry() calls interleave cannot both run
            # the body (which would create two uploaders and two heartbeats).
            self._initialized = True

            self._enabled = True
            self._recipe_only_ci_telemetry = False

            if self._disabled or is_telemetry_disabled_by_environment():
                type(self)._process_disabled = True
                self._disabled = True
                self._enabled = False
                return

            self._recipe_only_ci_telemetry = is_ci_environment()
            self._app_session_guid = str(uuid.uuid4())

            try:
                options = OneCollectorExporterOptions(
                    connection_string=base64.b64decode(
                        "SW5zdHJ1bWVudGF0aW9uS2V5PTYyMTUwOTExZGMwMDRmYzliYjY3YmE5NjA2NDI3ZTU2LWVjNjFmOWFmLTVkN2EtNGQxOS1hZjMxLWI5Y2Q2OWU5ODdmMS02OTE1"
                    ).decode()
                )
                options.validate()
                self._instrumentation_key = options.instrumentation_key
                self._envelope_ikey = (
                    f"{CommonSchemaJsonSerializationHelper.ONE_COLLECTOR_TENANCY_SYMBOL}:{options.tenant_token}"
                )

                # Durable on-disk queue + background uploader. The uploader
                # retries detailed events until delivery. CI has a separate
                # recipe-only queue so it cannot drain local action/error rows.
                db_file_name = CI_DB_FILE_NAME if self._recipe_only_ci_telemetry else DB_FILE_NAME
                db_path = os.path.join(get_telemetry_base_dir(), db_file_name)
                self._store = OfflineEventStore(db_path)
                if not self._store.is_open:
                    self._store = None
                    self._enabled = False
                    self._initialized = False
                    return
                self._uploader = EventUploader(self._store, instrumentation_key=self._instrumentation_key)
                if not self._recipe_only_ci_telemetry:
                    self._enqueue_heartbeat_once()
                    self._uploader.start()
            except Exception:
                # Fail silently — telemetry must never crash the host application
                if self._store is not None:
                    self._store.close()
                self._store = None
                self._uploader = None
                self._enabled = False
                self._initialized = False

    def _enqueue_heartbeat_once(self) -> None:
        """Reserve, enrich, and release this process's durable Heartbeat."""
        if type(self)._heartbeat_enqueued or self._store is None:
            return
        try:
            device_id, device_id_status = get_hashed_device_id_and_status()
            minimal_payload = self._build_payload(
                HEARTBEAT_EVENT_NAME,
                {
                    "device_id": device_id,
                    "device_id_status": device_id_status.value,
                },
            )
            if minimal_payload is None:
                return
            row_id = self._store.reserve(minimal_payload, _HEARTBEAT_RELEASE_SECONDS)
            if row_id is None:
                return
            type(self)._heartbeat_enqueued = True
            try:
                full_payload = self._build_payload(
                    HEARTBEAT_EVENT_NAME,
                    {
                        "device_id": device_id,
                        "device_id_status": device_id_status.value,
                        "os": platform.system(),
                        "os_version": platform.version(),
                        "os_release": platform.release(),
                        "os_arch": platform.machine(),
                    },
                )
                if full_payload is not None and self._store.release(row_id, full_payload):
                    return
            except Exception:
                pass
            # If enrichment fails, release the already-durable minimal event.
            self._store.release(row_id)
        except Exception:
            pass

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
            with self._lock:
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
        if filtered is None or not filtered:
            # Unknown/empty event: not whitelisted.
            return None
        event_data = dict(filtered)
        event_data.update(
            {
                "appName": "Olive",
                "LibraryVersion": VERSION,
                "AppSessionGuid": self._app_session_guid,
            }
        )
        serialized_snapshots = {}
        for snapshot_field in ("configOverrides", "packageConfigOverrides"):
            snapshot = event_data.get(snapshot_field)
            if not isinstance(snapshot, str):
                continue
            event_data.pop(snapshot_field)
            try:
                parsed_snapshot = json.loads(snapshot)
            except (TypeError, ValueError):
                continue
            scrubbed_snapshot = scrub_config_snapshot_for_telemetry(parsed_snapshot)
            serialized_snapshot = json.dumps(
                scrubbed_snapshot,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if len(serialized_snapshot.encode("utf-8")) > MAX_TELEMETRY_STRING_LENGTH:
                serialized_snapshot = '{"truncated":"[truncated]"}'
            serialized_snapshots[snapshot_field] = serialized_snapshot
        event_data = {
            key: value
            for key, value in event_data.items()
            if value is not None and isinstance(value, (str, bytes, bytearray, bool, int, float, datetime, os.PathLike))
        }
        scrubbed = scrub_value_for_telemetry(event_data)
        if not isinstance(scrubbed, dict):
            return None
        scrubbed.update(serialized_snapshots)
        exception_message = event_data.get("exceptionMessage")
        if event_name == ERROR_EVENT_NAME and isinstance(exception_message, str):
            scrubbed["exceptionMessage"] = scrub_error_message_for_telemetry(exception_message)
        envelope = CommonSchemaJsonSerializationHelper.create_event_envelope(
            event_name=event_name,
            timestamp=datetime.now(timezone.utc),
            ikey=self._envelope_ikey,
            data=scrubbed,
        )
        return CommonSchemaJsonSerializationHelper.serialize_to_json_bytes(envelope)

    def disable_telemetry(self) -> None:
        """Fully disable telemetry for the remainder of this process."""
        with self._lock:
            type(self)._process_disabled = True
            self._disabled = True
            self._enabled = False
            uploader_stopped = True
            if self._uploader is not None:
                self._uploader.retain_queued_rows()
                uploader_stopped = self._uploader.stop_loop(0)
            if self._uploader is not None and uploader_stopped:
                self._uploader.close()
                self._uploader = None
            if self._uploader is None and self._store is not None:
                self._store.close()
                self._store = None

    @classmethod
    def disable_process_telemetry(cls) -> None:
        """Latch full suppression without constructing telemetry resources."""
        with cls._lock:
            cls._process_disabled = True
            if cls._instance is not None:
                cls._instance.disable_telemetry()

    @classmethod
    def is_process_telemetry_disabled(cls) -> bool:
        """Return whether environment or API state fully disables this process."""
        with cls._lock:
            return cls._process_disabled or is_telemetry_disabled_by_environment()

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
            disabled = bool(getattr(self, "_disabled", False))
            if not disabled and bool(getattr(self, "_recipe_only_ci_telemetry", False)):
                flush_seconds = max(flush_seconds, callback_timeout_seconds)
            deadline = time.monotonic() + max(timeout_seconds, callback_timeout_seconds, flush_seconds)

            def remaining_seconds() -> float:
                return max(0.0, deadline - time.monotonic())

            uploader_stopped = True
            if self._uploader is not None:
                uploader_stopped = self._uploader.stop_loop(
                    join_timeout_seconds=0 if disabled else min(timeout_seconds, remaining_seconds())
                )
                if uploader_stopped:
                    if flush_seconds > 0 and not disabled:
                        flush_timeout = min(flush_seconds, remaining_seconds())
                        if flush_timeout > 0:
                            self._uploader.flush(flush_timeout)
                    self._uploader.close()
                    self._uploader = None
            if self._store is not None and uploader_stopped:
                self._store.close()
                self._store = None
            if self._uploader is None and self._store is None:
                self._initialized = False
        except Exception:
            # Fail silently — telemetry must never crash the host application
            pass

    def __del__(self):
        """Safety-net cleanup on garbage collection."""
        try:
            self.shutdown(timeout_millis=0, callback_timeout_millis=0, flush_seconds=0)
        except Exception:
            pass


def _get_logger() -> Optional[Telemetry]:
    """Get or create telemetry without publishing a disabled singleton."""
    return Telemetry.get_or_create_if_enabled()


def disable_telemetry() -> None:
    """Fully disable telemetry for the remainder of this process."""
    Telemetry.disable_process_telemetry()


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
        _set_nested_value(filtered, FIELD_NAMES.get(key, key), value)
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
