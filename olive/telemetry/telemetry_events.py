# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from datetime import datetime

from olive.telemetry.telemetry_logger import TelemetryLogger
from olive.telemetry.utils import _format_exception_msg

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


def log_error(action_name: str, called_from: str, error: Exception):
    logger.log(
        "OliveError",
        {
            "action_name": action_name,
            "called_from": called_from,
            "errorType": type(error).__name__,
            "error": _format_exception_msg(error),
        },
    )
