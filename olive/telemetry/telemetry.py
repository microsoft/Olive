# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Thin wrapper around the OneCollector telemetry logger with event helpers."""

import base64
import json
import pickle
import platform
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

from olive.telemetry.constants import CONNECTION_STRING
from olive.telemetry.library.event_source import event_source
from olive.telemetry.library.telemetry_logger import TelemetryLogger as _LibraryTelemetryLogger
from olive.telemetry.library.telemetry_logger import get_telemetry_logger
from olive.telemetry.deviceid import get_encrypted_device_id_and_status
from olive.telemetry.utils import get_telemetry_base_dir

# Default event names used by the high-level telemetry helpers.
HEARTBEAT_EVENT_NAME = "OliveHeartbeat"
ACTION_EVENT_NAME = "OliveAction"
ERROR_EVENT_NAME = "OliveError"

ALLOWED_KEYS = {
    HEARTBEAT_EVENT_NAME: [
        "device_id",
        "id_status",
        "os.name",
        "os.version",
        "os.release",
        "os.arch",
        "app_version",
        "app_instance_id",
    ],
    ACTION_EVENT_NAME: [
        "invoked_from",
        "action_name",
        "duration_ms",
        "success",
        "app_version",
        "app_instance_id",
    ],
    ERROR_EVENT_NAME: [
        "exception_type",
        "exception_message",
        "app_version",
        "app_instance_id",
    ],
}

CRITICAL_EVENTS = {HEARTBEAT_EVENT_NAME}
MAX_CACHE_SIZE_BYTES = 5 * 1024 * 1024
HARD_MAX_CACHE_SIZE_BYTES = 10 * 1024 * 1024
CACHE_FILE_NAME = "olive.pkl"


class TelemetryCacheHandler:
    def __init__(self, telemetry: "Telemetry") -> None:
        self._telemetry = telemetry
        self._cache_lock = threading.Lock()
        self._cache_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="olive_telemetry_cache")
        self._flush_in_progress = False
        self._transport = None

        # Track replayed events that haven't been sent yet
        self._pending_replay_events = []  # List of events being replayed
        self._pending_replay_lock = threading.Lock()
        self._replay_in_progress = False
        self._replay_writeback_scheduled = False
        self._replay_new_writes = False

    def setup_payload_callbacks(self) -> None:
        logger = self._telemetry._logger
        if not logger:
            return

        exporter = getattr(logger, "_logger_exporter", None)
        if not exporter:
            return

        transport = getattr(exporter, "_transport", None)
        if not transport:
            return
        if getattr(transport, "_cache_wrap_installed", False):
            return

        original_send = transport.send

        def wrapped_send(payload: bytes, timeout_sec: float, item_count: int = 1):
            transport._last_payload = payload
            return original_send(payload, timeout_sec, item_count=item_count)

        transport.send = wrapped_send
        transport._cache_wrap_installed = True
        self._transport = transport

        exporter.register_payload_transmitted_callback(self._on_payload_transmitted, include_failures=True)

    def shutdown(self) -> None:
        if self._cache_executor:
            # Wait for pending cache tasks to complete before shutting down
            # Note: We intentionally do NOT flush the cache here. The cache persists across
            # sessions for offline resilience. If network is working, the success callback
            # already flushed. If network is down, flushing would fail anyway.
            self._cache_executor.shutdown(wait=True)
            self._cache_executor = None

    def _on_payload_transmitted(self, args) -> None:
        try:
            if args.succeeded:
                # Telemetry succeeded - mark any pending replayed events as sent
                with self._pending_replay_lock:
                    if self._pending_replay_events:
                        # Remove events from pending list (they were successfully sent)
                        sent_count = min(len(self._pending_replay_events), args.item_count)
                        self._pending_replay_events = self._pending_replay_events[sent_count:]

                        # If no more pending replayed events, signal completion
                        if not self._pending_replay_events and self._replay_in_progress:
                            self._replay_in_progress = False
                            if not self._replay_new_writes and not self._replay_writeback_scheduled:
                                cache_path = self._get_cache_path()
                                if cache_path:
                                    cache_path.unlink(missing_ok=True)

                # Also flush any previously cached failures
                self._schedule_cache_task(self._flush_cache)
            else:
                # Telemetry failed - cache this payload for later replay
                with self._pending_replay_lock:
                    has_pending_replay = bool(self._pending_replay_events)
                    should_writeback = self._replay_in_progress and has_pending_replay
                    if should_writeback and not self._replay_writeback_scheduled:
                        self._replay_writeback_scheduled = True
                        self._schedule_cache_task(self._write_entries_to_cache, self._pending_replay_events.copy())
                payload = getattr(self._transport, "_last_payload", None)
                if payload:
                    self._schedule_cache_task(self._write_payload_to_cache, payload)
        except Exception:
            # Fail silently.
            pass

    def _schedule_cache_task(self, func, *args) -> None:
        try:
            if self._cache_executor:
                self._cache_executor.submit(func, *args)
            else:
                # If executor is not available (e.g., during shutdown), execute synchronously
                func(*args)
        except Exception:
            # Fail silently.
            pass

    def _get_telemetry_support_dir(self) -> Optional[Path]:
        return get_telemetry_base_dir()

    def _get_cache_path(self) -> Optional[Path]:
        support_dir = self._get_telemetry_support_dir()
        if not support_dir:
            return None
        return support_dir / "cache" / CACHE_FILE_NAME

    def _write_payload_to_cache(self, payload: bytes) -> None:
        try:
            cache_path = self._get_cache_path()
            if cache_path is None:
                return

            with self._pending_replay_lock:
                if self._replay_in_progress:
                    self._replay_new_writes = True

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_size = cache_path.stat().st_size if cache_path.exists() else 0

            if cache_size >= HARD_MAX_CACHE_SIZE_BYTES:
                return

            entries = _parse_payload(payload)
            if not entries:
                return

            if cache_size >= MAX_CACHE_SIZE_BYTES:
                entries = [entry for entry in entries if entry.get("event_name") in CRITICAL_EVENTS]
                if not entries:
                    return

            with self._cache_lock, cache_path.open("ab") as cache_file:
                for entry in entries:
                    pickle.dump(entry, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return

    def _write_entries_to_cache(self, entries: list[dict[str, Any]]) -> None:
        try:
            cache_path = self._get_cache_path()
            if cache_path is None:
                return

            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_size = cache_path.stat().st_size if cache_path.exists() else 0

            if cache_size >= HARD_MAX_CACHE_SIZE_BYTES:
                return

            if cache_size >= MAX_CACHE_SIZE_BYTES:
                entries = [entry for entry in entries if entry.get("event_name") in CRITICAL_EVENTS]
                if not entries:
                    return

            with self._cache_lock, cache_path.open("ab") as cache_file:
                for entry in entries:
                    pickle.dump(entry, cache_file, protocol=pickle.HIGHEST_PROTOCOL)
        except Exception:
            return

    def _flush_cache(self) -> None:
        with self._cache_lock:
            if self._flush_in_progress:
                return
            self._flush_in_progress = True

            try:
                cache_path = self._get_cache_path()
                if cache_path is None:
                    return
                if not cache_path.exists():
                    return

                entries = _read_cache_entries(cache_path)

                if not entries:
                    cache_path.unlink(missing_ok=True)
                    return

                cache_path.unlink(missing_ok=True)

                # Mark these entries as pending replay
                with self._pending_replay_lock:
                    self._pending_replay_events = entries.copy()
                    self._replay_in_progress = True
                    self._replay_writeback_scheduled = False
                    self._replay_new_writes = False

                for entry in entries:
                    try:
                        event_name = entry.get("event_name")
                        event_data = entry.get("event_data")
                        if not event_name or not event_data:
                            continue
                        attributes = json.loads(event_data)
                        if not isinstance(attributes, dict):
                            continue
                        attributes["initTs"] = entry.get("ts")
                        self._telemetry.log(event_name, attributes, None)
                    except Exception:
                        # Remove failed entry from pending list
                        with self._pending_replay_lock:
                            if entry in self._pending_replay_events:
                                self._pending_replay_events.remove(entry)
                        continue

                self._telemetry.force_flush(timeout_millis=5_000)
            except Exception:
                return
            finally:
                self._flush_in_progress = False


class Telemetry:
    """Wrapper that wires environment configuration into the library logger."""

    _instance: Optional["Telemetry"] = None
    _initialized: bool = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._logger = self._create_logger()
        self._cache_handler = TelemetryCacheHandler(self)
        self._initialized = True
        self._setup_payload_callbacks()
        self._log_heartbeat()
        event_source.disable()

    def _create_logger(self) -> Optional[_LibraryTelemetryLogger]:
        try:
            return get_telemetry_logger(base64.b64decode(CONNECTION_STRING).decode())
        except Exception:
            return None

    def _setup_payload_callbacks(self) -> None:
        if not self._logger:
            return
        self._cache_handler.setup_payload_callbacks()

    def add_global_metadata(self, metadata: dict[str, Any]) -> None:
        if self._logger:
            self._logger.add_global_metadata(metadata)

    def log(
        self, event_name: str, attributes: Optional[dict[str, Any]] = None, metadata: Optional[dict[str, Any]] = None
    ) -> None:
        if self._logger:
            attrs = _merge_metadata(attributes, metadata)
            self._logger.log(event_name, attrs)

    def _log_heartbeat(
        self,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Log a heartbeat event with system information.

        Args:
            metadata: Optional additional metadata to include.

        """
        encrypted_device_id, device_id_status = get_encrypted_device_id_and_status()
        attributes = {
            "device_id": encrypted_device_id,
            "id_status": device_id_status.value,
            "os": {
                "name": platform.system().lower(),
                "version": platform.version(),
                "release": platform.release(),
                "arch": platform.machine(),
            },
        }
        self.log(HEARTBEAT_EVENT_NAME, attributes, metadata)

    def disable_telemetry(self) -> None:
        if self._logger:
            self._logger.disable_telemetry()

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        if self._logger and hasattr(self._logger, "force_flush"):
            return self._logger.force_flush(timeout_millis=timeout_millis)
        return False

    def shutdown(self) -> None:
        # Shutdown cache handler FIRST to ensure pending cache tasks complete
        # The cache handler will wait for replayed events to be sent before returning
        if self._cache_handler:
            self._cache_handler.shutdown()

        if self._logger:
            self._logger.shutdown()


def _get_logger() -> Telemetry:
    """Get or create the singleton Telemetry instance."""
    return Telemetry()


def _merge_metadata(attributes: Optional[dict[str, Any]], metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    merged = dict(attributes or {})
    if metadata:
        merged.update(metadata)
    return merged


def _parse_payload(payload: bytes) -> list[dict[str, Any]]:
    entries = []
    try:
        payload_text = payload.decode("utf-8")
        lines = payload_text.splitlines()

        for raw_line in lines:
            line = raw_line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
                event_name = event.get("name")
                if not event_name:
                    continue
                filtered_data = _filter_event_data(event_name, event.get("data") or {})
                if not filtered_data:
                    continue
                entries.append(
                    {
                        "ts": event.get("time") or time.time(),
                        "event_name": event_name,
                        "event_data": json.dumps(filtered_data, ensure_ascii=False, separators=(",", ":")),
                    }
                )
            except Exception:
                continue
    except Exception:
        return []

    return entries


def _filter_event_data(event_name: str, data: dict[str, Any]) -> Optional[dict[str, Any]]:
    allowed_keys = ALLOWED_KEYS.get(event_name)
    if not allowed_keys:
        return None

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


def _read_cache_entries(cache_path: Path) -> list[dict[str, Any]]:
    entries = []
    try:
        with cache_path.open("rb") as cache_file:
            while True:
                try:
                    entry = pickle.load(cache_file)
                    entries.append(entry)
                except EOFError:
                    break
                except Exception:
                    continue
    except Exception:
        return []
    return entries
