# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from olive.telemetry.library.msft_log_exporter import MSFTLogExporter


class TelemetryLogger:
    _instance = None  # Class-level attribute to store the single instance
    _logger: logging.Logger = None
    _logger_provider: LoggerProvider = None

    def __new__(cls, *args, **kwargs):
        # Check if an instance already exists
        if cls._instance is None:
            # If not, create a new instance and store it in _instance
            cls._instance = super().__new__(cls)

            try:
                exporter = MSFTLogExporter()
                cls._logger_provider = LoggerProvider(
                    resource=Resource.create(
                        {
                            "service.name": "olive-telemetry",
                            "service.instance.id": "olive-telemetry-instance",
                        }
                    ),
                )
                set_logger_provider(cls._logger_provider)
                cls._logger_provider.add_log_record_processor(BatchLogRecordProcessor(exporter))
                handler = LoggingHandler(level=logging.INFO, logger_provider=cls._logger_provider)

                logger = logging.getLogger("olive.telemetry")
                logger.propagate = False
                logger.setLevel(logging.INFO)
                logger.addHandler(handler)
                cls._logger = logger
            except Exception:
                # If any error occurs during initialization, we will not set up the logger and will silently fail.
                cls._logger = None
                cls._logger_provider = None
        return cls._instance

    def __init__(self):
        pass

    def log(self, event_name: str, information: dict[str, Any]):
        if self._logger:  # in case the logger was not initialized properly
            self._logger.info(event_name, extra=information)

    def disable_telemetry(self):
        if self._logger:  # in case the logger was not initialized properly
            self._logger.disabled = True

    def shutdown(self):
        if self._logger_provider:  # in case the logger provider was not initialized properly
            self._logger_provider.shutdown()
