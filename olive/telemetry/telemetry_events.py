# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import inspect
import traceback
from datetime import datetime

from olive.telemetry.telemetry_logger import TelemetryLogger


def action(func):
    t = TelemetryLogger()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        stack = inspect.stack()
        caller_frame = stack[1]
        caller_module = inspect.getmodule(caller_frame[0])
        called_from = caller_module.__name__

        if caller_module is None:
            called_from = "Interactive"
        elif caller_module.__name__ == "__main__":
            called_from = "Script"

        success = False
        error = None
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as ex:
            result = None
            error = ex
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        action_name = args[0].__class__.__name__ if args else "Invalid"
        if action_name.endswith("Command"):
            action_name = action_name[: -len("Command")]
        t.log(
            "OliveAction",
            {
                "action": action_name,
                "caller": called_from,
                "actionTime": start_time,
                "timeMs": duration_ms,
                "success": success,
            },
        )

        if error:
            t.log(
                "OliveError",
                {
                    "action": action_name,
                    "caller": called_from,
                    "errorType": type(error).__name__,
                    "error": _format_exception_msg(error),
                },
            )
            raise error
        return result

    return wrapper


def _format_exception_msg(exc: Exception) -> str:
    folder = "Olive"
    file_line = 'File "'
    exc = traceback.format_exception(exc, limit=5)
    lines = []
    for line in exc:
        line_trunc = line.strip()
        if line_trunc.startswith(file_line) and folder in line_trunc:
            idx = line_trunc.find(folder)
            if idx != -1:
                line_trunc = line_trunc[idx + len(folder) :]
        elif line_trunc.startswith(file_line):
            idx = line_trunc[len(file_line) :].find('"')
            line_trunc = line_trunc[idx + len(file_line) :]
        lines.append(line_trunc)
    return "\n".join(lines)
