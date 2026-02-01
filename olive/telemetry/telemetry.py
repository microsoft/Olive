# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Thin wrapper around the OneCollector telemetry logger with event helpers."""

import base64
import platform
import uuid
from typing import Any, Optional

from olive.telemetry.constants import CONNECTION_STRING
from olive.telemetry.library.telemetry_logger import TelemetryLogger as _LibraryTelemetryLogger
from olive.telemetry.library.telemetry_logger import get_telemetry_logger
from olive.telemetry.utils import _generate_encrypted_device_id
from olive.version import __version__ as VERSION

# Default event names used by the high-level telemetry helpers.
HEARTBEAT_EVENT_NAME = "OliveHeartbeat"
ACTION_EVENT_NAME = "OliveAction"
ERROR_EVENT_NAME = "OliveError"


class Telemetry:
    """Wrapper that wires environment configuration into the library logger."""

    _instance: Optional["Telemetry"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._logger = self._create_logger()
        self._session_id = str(uuid.uuid4())
        self._initialized = True

    def _create_logger(self) -> Optional[_LibraryTelemetryLogger]:
        try:
            return get_telemetry_logger(base64.b64decode(CONNECTION_STRING).decode())
        except Exception:
            return None

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        if self._logger:
            self._logger.add_metadata(metadata)

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        if self._logger:
            # Always include session_id in every event
            attrs = dict(attributes or {})
            attrs["session_id"] = self._session_id
            self._logger.log(event_name, attrs)

    def disable_telemetry(self) -> None:
        if self._logger:
            self._logger.disable_telemetry()

    def shutdown(self) -> None:
        if self._logger:
            self._logger.shutdown()


def _get_logger() -> Telemetry:
    """Get or create the singleton Telemetry instance."""
    return Telemetry()


def _merge_metadata(attributes: dict[str, Any], metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    if metadata:
        return {**attributes, **metadata}
    return attributes


def log_heartbeat(
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log a heartbeat event with system information.

    Args:
        metadata: Optional additional metadata to include.

    """
    logger = _get_logger()
    attributes = {
        "device_id": _generate_encrypted_device_id(),
        "os": {
            "name": platform.system().lower(),
            "version": platform.version(),
            "release": platform.release(),
            "arch": platform.machine(),
        },
        "version": VERSION,
    }
    logger.log(HEARTBEAT_EVENT_NAME, _merge_metadata(attributes, metadata))


def log_action(
    invoked_from: str,
    action_name: str,
    duration_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log an action event.

    Args:
        invoked_from: Where the action was invoked from.
        action_name: Name of the action.
        duration_ms: Duration in milliseconds.
        success: Whether the action succeeded.
        metadata: Optional additional metadata to include.

    """
    logger = _get_logger()
    attributes = {
        "invoked_from": invoked_from,
        "action_name": action_name,
        "duration_ms": duration_ms,
        "success": success,
    }
    logger.log(ACTION_EVENT_NAME, _merge_metadata(attributes, metadata))


def log_error(
    invoked_from: str,
    exception_type: str,
    exception_message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    """Log an error event.

    Args:
        invoked_from: Where the error occurred.
        exception_type: Type of the exception.
        exception_message: Exception message.
        metadata: Optional additional metadata to include.

    """
    logger = _get_logger()
    attributes = {
        "invoked_from": invoked_from,
        "exception_type": exception_type,
        "exception_message": exception_message,
    }
    logger.log(ERROR_EVENT_NAME, _merge_metadata(attributes, metadata))
