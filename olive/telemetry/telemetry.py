import functools
import inspect
from datetime import datetime

from olive.telemetry.telemetry_events import log_action, log_error
from olive.telemetry.utils import _format_exception_msg


def action(func):
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
        exception = None
        start_time = datetime.now()
        try:
            result = func(*args, **kwargs)
            success = True
        except Exception as ex:
            result = None
            exception = ex
        duration_ms = int((datetime.now() - start_time).total_seconds() * 1000)
        action_name = args[0].__class__.__name__ if args else "Invalid"

        if action_name.endswith("Command"):
            action_name = action_name[: -len("Command")]

        log_action(action_name, called_from, start_time, duration_ms, success)

        if exception:
            exception_type = type(exception).__name__
            exception_message = _format_exception_msg(exception)
            log_error(action_name, called_from, exception_type, exception_message)
            raise exception

        return result

    return wrapper
