# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import inspect
import time
import traceback
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

from olive.telemetry.telemetry import ACTION_EVENT_NAME, ERROR_EVENT_NAME, RECIPE_EVENT_NAME, _get_logger
from olive.telemetry.telemetry_redaction import scrub_error_message_for_telemetry, scrub_string_for_telemetry

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])
_ERROR_LOGGED_ATTR = "_olive_telemetry_logged"


def _scrub_metadata_value(value):
    if isinstance(value, str):
        return _redact_paths(value)
    if isinstance(value, dict):
        return {
            _redact_paths(key) if isinstance(key, str) else key: _scrub_metadata_value(child)
            for key, child in value.items()
        }
    if isinstance(value, list):
        return [_scrub_metadata_value(child) for child in value]
    if isinstance(value, tuple):
        return tuple(_scrub_metadata_value(child) for child in value)
    return value


def _scrub_metadata(metadata: Optional[dict[str, Any]]) -> dict[str, Any]:
    return _scrub_metadata_value(metadata or {})


def log_action(
    invoked_from: str,
    action_name: str,
    duration_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "invoked_from": invoked_from,
        "action_name": action_name,
        "duration_ms": duration_ms,
        "success": success,
    }
    telemetry.log(ACTION_EVENT_NAME, attributes, _scrub_metadata(metadata))


def log_error(
    exception_type: str,
    exception_message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "exception_type": exception_type,
        "exception_message": _redact_error_message(exception_message),
    }
    telemetry.log(ERROR_EVENT_NAME, attributes, _scrub_metadata(metadata))


def log_recipe_result(
    recipe_name: str,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "recipe_name": recipe_name,
        "success": success,
    }
    telemetry.log(RECIPE_EVENT_NAME, attributes, metadata)


def _redact_paths(text: str) -> str:
    return scrub_string_for_telemetry(text)


def _redact_error_message(text: str) -> str:
    return scrub_error_message_for_telemetry(text)


def _is_exception_logged(exc: BaseException) -> bool:
    return bool(getattr(exc, _ERROR_LOGGED_ATTR, False))


def _mark_exception_logged(exc: BaseException) -> None:
    try:
        setattr(exc, _ERROR_LOGGED_ATTR, True)
    except Exception:
        # Some exception implementations do not allow custom attributes.
        pass


def _format_exception_message(ex: BaseException, tb: Optional[TracebackType] = None) -> str:
    """Format an exception and strip local paths for privacy.

    Each entry from ``traceback.format_exception`` is a multi-line string (the
    ``File "..."`` line plus the offending source line), so we process every
    physical line: filenames are replaced with ``[path]``, and any path that
    remains on a source or message line is redacted so a username embedded in it
    cannot leak into OliveError.
    """
    file_line = 'File "'
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for chunk in formatted:
        for raw_line in chunk.splitlines():
            line_trunc = raw_line.strip()
            if line_trunc.startswith(file_line):
                path_end = line_trunc.find('"', len(file_line))
                if path_end != -1:
                    line_trunc = f'File "[path]"{line_trunc[path_end + 1 :]}'
            line_trunc = _redact_error_message(line_trunc)
            lines.append(line_trunc)
    return "\n".join(lines)


def _resolve_invoked_from(skip_frames: int = 0) -> str:
    """Resolve how Olive was invoked by examining the call stack.

    Walks up the stack to find the first frame outside the olive package,
    which indicates how the user invoked Olive (CLI, script, interactive, etc.).

    :param skip_frames: Number of additional frames to skip (for internal use).
    :return: A string indicating how Olive was invoked.
    """
    for frame_info in inspect.stack()[2 + skip_frames :]:  # skip this function and caller
        module = inspect.getmodule(frame_info.frame)
        if module is None:
            # Could be interactive or dynamically generated code
            continue
        module_name = module.__name__
        # Skip olive internals to find user code
        if module_name.startswith("olive."):
            continue
        if module_name == "__main__":
            return "Script"
        return module_name
    return "Interactive"


def _resolve_action_name(func: Callable[..., Any]) -> str:
    action_name = getattr(func, "__name__", "unknown")
    qualname = getattr(func, "__qualname__", action_name)
    if "." not in qualname:
        return action_name
    owner = qualname.rsplit(".", 1)[0].rsplit(".", 1)[-1]
    if owner == "<locals>":
        return action_name
    owner = owner[: -len("Command")] if owner.endswith("Command") else owner
    return owner if action_name == "run" else f"{owner}.{action_name}"


class ActionContext:
    """Context manager for recording telemetry around a block of work."""

    def __init__(
        self,
        action_name: str,
        invoked_from: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.action_name = action_name
        try:
            self._telemetry_enabled = _get_logger().accepts_detailed_events
        except Exception:
            self._telemetry_enabled = False
        self.invoked_from = (
            invoked_from
            if invoked_from is not None
            else _resolve_invoked_from()
            if self._telemetry_enabled
            else "disabled"
        )
        self.metadata = metadata or {}
        self._start_time: Optional[float] = None

    def add_metadata(self, key: str, value: Any) -> None:
        self.metadata[key] = value

    def __enter__(self) -> "ActionContext":
        self._start_time = time.perf_counter()
        return self

    def __exit__(
        self,
        exc_type: Optional[type[BaseException]],
        exc_val: Optional[BaseException],
        exc_tb: Optional[TracebackType],
    ) -> bool:
        if not self._telemetry_enabled:
            return False
        end_time = time.perf_counter()
        start_time = self._start_time if self._start_time is not None else end_time
        duration_ms = int((end_time - start_time) * 1000)
        success = exc_type is None

        log_action(
            invoked_from=self.invoked_from,
            action_name=self.action_name,
            duration_ms=duration_ms,
            success=success,
            metadata=self.metadata,
        )

        if exc_type is not None and exc_val is not None and not _is_exception_logged(exc_val):
            log_error(
                exception_type=exc_type.__name__,
                exception_message=_format_exception_message(exc_val, exc_tb),
                metadata=self.metadata,
            )
            _mark_exception_logged(exc_val)

        # Do not suppress exceptions
        return False


def action(func: _TFunc) -> _TFunc:
    """Record telemetry around a function call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            if not _get_logger().accepts_detailed_events:
                return func(*args, **kwargs)
        except Exception:
            return func(*args, **kwargs)

        # Resolve telemetry context defensively: instrumentation (including
        # inspect.stack()) must never propagate into the wrapped call.
        try:
            invoked_from = _resolve_invoked_from()
            action_name = _resolve_action_name(func)
        except Exception:
            invoked_from = "unknown"
            action_name = getattr(func, "__name__", "unknown")

        start_time = time.perf_counter()
        success = True
        try:
            return func(*args, **kwargs)
        except Exception as exc:
            success = False
            if not _is_exception_logged(exc):
                log_error(
                    exception_type=type(exc).__name__,
                    exception_message=_format_exception_message(exc, exc.__traceback__),
                )
                _mark_exception_logged(exc)
            raise
        finally:
            duration_ms = int((time.perf_counter() - start_time) * 1000)
            log_action(
                invoked_from=invoked_from,
                action_name=action_name,
                duration_ms=duration_ms,
                success=success,
            )

    return wrapper  # type: ignore[return-value]
