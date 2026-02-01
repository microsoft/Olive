# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""High-level telemetry logger facade for easy usage."""

import logging
from typing import Any, Optional

from opentelemetry._logs import set_logger_provider
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor
from opentelemetry.sdk.resources import Resource

from olive.telemetry.library.exporter import OneCollectorLogExporter
from olive.telemetry.library.options import OneCollectorExporterOptions


class TelemetryLogger:
    """Singleton telemetry logger for simplified OneCollector integration.

    Provides a simple interface for logging telemetry events without
    needing to configure OpenTelemetry directly.
    """

    _instance: Optional["TelemetryLogger"] = None
    _logger: Optional[logging.Logger] = None
    _logger_exporter: Optional[OneCollectorLogExporter] = None
    _logger_provider: Optional[LoggerProvider] = None

    def __new__(cls, options: Optional[OneCollectorExporterOptions] = None):
        """Create or return the singleton instance.

        Args:
            options: Exporter options (only used on first instantiation)

        """
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialize(options)

        return cls._instance

    def _initialize(self, options: Optional[OneCollectorExporterOptions]) -> None:
        """Initialize the logger (called only once).

        Args:
            options: Exporter configuration options

        """
        try:
            # Create exporter
            self._logger_exporter = OneCollectorLogExporter(options=options)

            # Create logger provider
            self._logger_provider = LoggerProvider(
                resource=Resource.create(
                    {
                        "service.name": "telemetry_logger",
                        "service.instance.id": "telemetry_logger-instance",
                    }
                )
            )

            # Set as global logger provider
            set_logger_provider(self._logger_provider)

            # Add batch processor
            self._logger_provider.add_log_record_processor(BatchLogRecordProcessor(self._logger_exporter))

            # Create logging handler
            handler = LoggingHandler(level=logging.INFO, logger_provider=self._logger_provider)

            # Set up Python logger
            logger = logging.getLogger(__name__)
            logger.propagate = False
            logger.setLevel(logging.INFO)
            logger.addHandler(handler)

            self._logger = logger

        except Exception:
            # Silently fail initialization - logger will be None
            self._logger = None
            self._logger_provider = None
            self._logger_exporter = None

    def add_metadata(self, metadata: dict[str, Any]) -> None:
        """Add metadata fields to all telemetry events.

        Args:
            metadata: Dictionary of metadata to add

        """
        if self._logger_exporter:
            self._logger_exporter.add_metadata(metadata)

    def log(self, event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
        """Log a telemetry event.

        Args:
            event_name: Name of the event
            attributes: Optional event attributes

        """
        if self._logger:
            extra = attributes if attributes else {}
            self._logger.info(event_name, extra=extra)

    def disable_telemetry(self) -> None:
        """Disable telemetry logging."""
        if self._logger:
            self._logger.disabled = True

    def enable_telemetry(self) -> None:
        """Enable telemetry logging."""
        if self._logger:
            self._logger.disabled = False

    def shutdown(self) -> None:
        """Shutdown the telemetry logger and flush pending data."""
        if self._logger_provider:
            self._logger_provider.shutdown()


# Convenience functions for common use cases
_default_logger: Optional[TelemetryLogger] = None


def get_telemetry_logger(connection_string: Optional[str] = None) -> TelemetryLogger:
    """Get or create the default telemetry logger.

    Args:
        connection_string: OneCollector connection string (only used on first call)

    Returns:
        TelemetryLogger instance

    """
    global _default_logger

    if _default_logger is None:
        options = None
        if connection_string:
            options = OneCollectorExporterOptions(connection_string=connection_string)
        _default_logger = TelemetryLogger(options=options)

    return _default_logger


def log_event(event_name: str, attributes: Optional[dict[str, Any]] = None) -> None:
    """Log a telemetry event using the default logger.

    Args:
        event_name: Name of the event
        attributes: Optional event attributes

    """
    logger = get_telemetry_logger()
    logger.log(event_name, attributes)


def shutdown_telemetry() -> None:
    """Shutdown the default telemetry logger."""
    global _default_logger
    if _default_logger:
        _default_logger.shutdown()
        _default_logger = None
