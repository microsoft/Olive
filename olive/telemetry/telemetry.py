import functools
import inspect
from datetime import datetime

from olive.telemetry.telemetry_events import log_action, log_error


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

        log_action(action_name, called_from, start_time, duration_ms, success)

        if error:
            log_error(action_name, called_from, error)
            raise error

        return result

    return wrapper
