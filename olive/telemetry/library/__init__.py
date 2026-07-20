# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""OneCollector building blocks (standard library only).

Helpers for serializing telemetry to Common Schema JSON and posting it to the
Microsoft OneCollector endpoint. These modules have no third-party dependency
and are driven directly by the SQLite-backed uploader.
"""

from olive.telemetry.library.callback_manager import CallbackManager, PayloadTransmittedCallbackArgs
from olive.telemetry.library.connection_string_parser import ConnectionStringParser
from olive.telemetry.library.event_source import OneCollectorEventId, OneCollectorEventSource, event_source
from olive.telemetry.library.options import (
    CompressionType,
    OneCollectorExporterOptions,
    OneCollectorExporterValidationError,
    OneCollectorTransportOptions,
)
from olive.telemetry.library.payload_builder import PayloadBuilder
from olive.telemetry.library.serialization import CommonSchemaJsonSerializationHelper
from olive.telemetry.library.transport import HttpJsonPostTransport, ITransport

__all__ = [
    "CallbackManager",
    "CommonSchemaJsonSerializationHelper",
    "CompressionType",
    "ConnectionStringParser",
    "HttpJsonPostTransport",
    "ITransport",
    "OneCollectorEventId",
    "OneCollectorEventSource",
    "OneCollectorExporterOptions",
    "OneCollectorExporterValidationError",
    "OneCollectorTransportOptions",
    "PayloadBuilder",
    "PayloadTransmittedCallbackArgs",
    "event_source",
]
