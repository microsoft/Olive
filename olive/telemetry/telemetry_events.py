# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.telemetry.library.telemetry_logger import TelemetryLogger

logger = TelemetryLogger()


def log_action(action_name: str, duration_ms: float, success: bool, called_from: str = "module"):
    logger.log(
        "OliveAction",
        {
            "action_name": action_name,
            "duration_ms": duration_ms,
            "success": success,
            "called_from": called_from,
        },
    )


def log_error(exception_type: str, exception_message: str, called_from: str = "module"):
    logger.log(
        "OliveError",
        {
            "exception_type": exception_type,
            "exception_message": exception_message,
            "called_from": called_from,
        },
    )
