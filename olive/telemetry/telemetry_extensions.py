# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import inspect
import re
import time
import traceback
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar

from olive.telemetry.telemetry import ACTION_EVENT_NAME, ERROR_EVENT_NAME, RECIPE_EVENT_NAME, _get_logger

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])
_ERROR_LOGGED_ATTR = "_olive_telemetry_logged"


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
    telemetry.log(ACTION_EVENT_NAME, attributes, metadata)


def log_error(
    exception_type: str,
    exception_message: str,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    attributes = {
        "exception_type": exception_type,
        "exception_message": _redact_paths(exception_message),
    }
    telemetry.log(ERROR_EVENT_NAME, attributes, metadata)


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
    """Redact path-bearing tails without leaking space-containing user names."""
    pattern = re.compile(
        r"(?:[A-Za-z]:[\\/])"
        r"|(?:\\\\)"
        r"|(?:~[\\/])"
        r"|(?:(?<![:/])/(?:[^/\r\n]+/))"
        r"|(?:(?<![\\/\w])(?:[A-Za-z0-9_.-]+\\)[^\\/\r\n]+\\)",
        re.IGNORECASE,
    )
    redacted = []
    for line in text.splitlines(keepends=True):
        body = line.rstrip("\r\n")
        ending = line[len(body) :]
        match = pattern.search(body)
        redacted.append(body[: match.start()] + "<path>" + ending if match else line)
    return "".join(redacted)


def _is_exception_logged(exc: BaseException) -> bool:
    return bool(getattr(exc, _ERROR_LOGGED_ATTR, False))


def _mark_exception_logged(exc: BaseException) -> None:
    try:
        setattr(exc, _ERROR_LOGGED_ATTR, True)
    except Exception:
        pass


def _format_exception_message(ex: BaseException, tb: Optional[TracebackType] = None) -> str:
    """Format an exception and strip local paths for privacy.

    Each entry from ``traceback.format_exception`` is a multi-line string (the
    ``File "..."`` line plus the offending source line), so we process every
    physical line: filenames are trimmed to a package-relative form, and any
    absolute path that remains on a source or message line is redacted so a
    username embedded in it cannot leak into OliveError.
    """
    folder = "Olive"
    file_line = 'File "'
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for chunk in formatted:
        for raw_line in chunk.splitlines():
            line_trunc = raw_line.strip()
            if line_trunc.startswith(file_line) and folder in line_trunc:
                idx = line_trunc.find(folder)
                if idx != -1:
                    line_trunc = line_trunc[idx + len(folder) :]
            elif line_trunc.startswith(file_line):
                idx = line_trunc[len(file_line) :].find('"')
                line_trunc = line_trunc[idx + len(file_line) :]
            # Redact any absolute path that remains (source lines, message, and
            # the tail of File lines).
            line_trunc = _redact_paths(line_trunc)
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


class ActionContext:
    """Context manager for recording telemetry around a block of work."""

    def __init__(
        self,
        action_name: str,
        invoked_from: Optional[str] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.action_name = action_name
        self.invoked_from = invoked_from if invoked_from is not None else _resolve_invoked_from()
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
        duration_ms = int((time.perf_counter() - (self._start_time or time.perf_counter())) * 1000)
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
        # Resolve telemetry context defensively: instrumentation (including
        # inspect.stack()) must never propagate into the wrapped call.
        try:
            invoked_from = _resolve_invoked_from()
            action_name = func.__name__
            if args and hasattr(args[0], "__class__"):
                cls_name = args[0].__class__.__name__
                cls_name = cls_name[: -len("Command")] if cls_name.endswith("Command") else cls_name
                if cls_name:
                    action_name = cls_name if action_name == "run" else f"{cls_name}.{action_name}"
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
