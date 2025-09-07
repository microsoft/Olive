# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from datetime import datetime

from olive.telemetry.telemetry_logger import TelemetryLogger

logger = TelemetryLogger()


def log_action(action_name: str, called_from: str, start_time: datetime, duration_ms: float, success: bool):
    logger.log(
        "OliveAction",
        {
            "action_name": action_name,
            "called_from": called_from,
            "start_time": start_time,
            "duration_ms": duration_ms,
            "success": success,
        },
    )


def log_error(action_name: str, called_from: str, exception_type: str, exception_message: str):
    logger.log(
        "OliveError",
        {
            "action_name": action_name,
            "called_from": called_from,
            "exception_type": exception_type,
            "exception_message": exception_message,
        },
    )
