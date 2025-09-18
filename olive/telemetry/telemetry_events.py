# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from olive.telemetry.library.telemetry_logger import TelemetryLogger

logger = TelemetryLogger()


def log_action(called_from: str, action_name: str, duration_ms: float, success: bool):
    logger.log(
        "OliveAction",
        {
            "called_from": called_from,
            "action_name": action_name,
            "duration_ms": duration_ms,
            "success": success,
        },
    )


def log_error(called_from: str, exception_type: str, exception_message: str):
    logger.log(
        "OliveError",
        {
            "called_from": called_from,
            "exception_type": exception_type,
            "exception_message": exception_message,
        },
    )
