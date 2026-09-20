# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import functools
import inspect
import time
import traceback
from collections import deque
from contextlib import suppress
from types import TracebackType
from typing import Any, Callable, Optional, TypeVar, cast

from olive.telemetry.telemetry import ACTION_EVENT_NAME, ERROR_EVENT_NAME, RECIPE_EVENT_NAME, _get_logger
from olive.telemetry.telemetry_redaction import scrub_error_message_for_telemetry

_TFunc = TypeVar("_TFunc", bound=Callable[..., Any])
_ERROR_LOGGED_ATTR = "_olive_telemetry_logged"


def log_action(
    invoked_from: str,
    action_name: str,
    duration_ms: float,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    with suppress(Exception):
        telemetry = _get_logger()
        if telemetry is None:
            return
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
    *,
    stack_trace: Optional[str] = None,
    inner_exception_type: Optional[str] = None,
    inner_exception_message: Optional[str] = None,
    inner_stack_trace: Optional[str] = None,
) -> None:
    with suppress(Exception):
        telemetry = _get_logger()
        if telemetry is None:
            return
        attributes = {
            "exception_type": exception_type,
            "exception_message": _redact_error_message(exception_message),
            "stack_trace": _redact_error_message(stack_trace) if stack_trace else None,
            "inner_exception_type": inner_exception_type,
            "inner_exception_message": (
                _redact_error_message(inner_exception_message) if inner_exception_message else None
            ),
            "inner_stack_trace": _redact_error_message(inner_stack_trace) if inner_stack_trace else None,
        }
        telemetry.log(ERROR_EVENT_NAME, attributes, metadata)


def log_recipe_result(
    recipe_name: str,
    success: bool,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    telemetry = _get_logger()
    if telemetry is None:
        return
    attributes = {
        "recipe_name": recipe_name,
        "success": success,
    }
    telemetry.log(RECIPE_EVENT_NAME, attributes, metadata)


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


def _get_exception_message(ex: BaseException) -> str:
    try:
        message = str(ex)
    except Exception:
        message = "<exception str() failed>"
    return _redact_error_message(message)


def _format_stack_trace(tb: Optional[TracebackType]) -> Optional[str]:
    """Format bounded frame metadata without collecting paths or source-code lines."""
    if tb is None:
        return None
    lines = []
    for frame, line_number in deque(traceback.walk_tb(tb), maxlen=5):
        frame_name = _redact_error_message(frame.f_code.co_name)
        lines.append(f'File "[path]", line {line_number}, in {frame_name}')
    if not lines:
        return None
    lines.insert(0, "Traceback (most recent call last):")
    return _redact_error_message("\n".join(lines))


def _get_inner_exception(ex: BaseException) -> Optional[BaseException]:
    if ex.__cause__ is not None:
        return ex.__cause__
    if not ex.__suppress_context__:
        return ex.__context__
    return None


def _build_exception_details(ex: BaseException, tb: Optional[TracebackType] = None) -> dict[str, Optional[str]]:
    details = {
        "exception_type": type(ex).__name__,
        "exception_message": _get_exception_message(ex),
        "stack_trace": _format_stack_trace(tb),
        "inner_exception_type": None,
        "inner_exception_message": None,
        "inner_stack_trace": None,
    }
    inner = _get_inner_exception(ex)
    if inner is not None:
        details.update(
            {
                "inner_exception_type": type(inner).__name__,
                "inner_exception_message": _get_exception_message(inner),
                "inner_stack_trace": _format_stack_trace(inner.__traceback__),
            }
        )
    return details


def _log_exception(
    ex: BaseException,
    tb: Optional[TracebackType] = None,
    metadata: Optional[dict[str, Any]] = None,
) -> None:
    with suppress(Exception):
        log_error(**_build_exception_details(ex, tb), metadata=metadata)


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
            telemetry = _get_logger()
            self._telemetry_enabled = bool(telemetry is not None and telemetry.accepts_detailed_events)
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
            _log_exception(exc_val, exc_tb, self.metadata)
            _mark_exception_logged(exc_val)

        # Do not suppress exceptions
        return False


def action(func: _TFunc) -> _TFunc:
    """Record telemetry around a function call."""

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any):
        try:
            telemetry = _get_logger()
            if telemetry is None or not telemetry.accepts_detailed_events:
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
                _log_exception(exc, exc.__traceback__)
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

    return cast("_TFunc", wrapper)
