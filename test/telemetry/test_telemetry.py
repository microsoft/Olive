# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock

from olive.telemetry.library.telemetry_logger import TelemetryLogger
from olive.telemetry.telemetry import Telemetry


def test_telemetry_logger_log_reports_if_event_was_submitted():
    telemetry_logger = object.__new__(TelemetryLogger)
    telemetry_logger._logger = MagicMock(disabled=True)

    assert not telemetry_logger.log("event")
    telemetry_logger._logger.info.assert_not_called()

    telemetry_logger._logger.disabled = False
    assert telemetry_logger.log("event", {"key": "value"})
    telemetry_logger._logger.info.assert_called_once_with("event", extra={"key": "value"})


def test_telemetry_records_only_submitted_events():
    telemetry = object.__new__(Telemetry)
    telemetry._logger = MagicMock()
    telemetry._cache_handler = MagicMock()

    telemetry._logger.log.return_value = False
    telemetry.log("disabled-event")
    telemetry._cache_handler.record_event_logged.assert_not_called()

    telemetry._logger.log.return_value = True
    telemetry.log("enabled-event")
    telemetry._cache_handler.record_event_logged.assert_called_once_with()
