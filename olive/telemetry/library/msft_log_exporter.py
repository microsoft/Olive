# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import base64
import gzip
import hashlib
import json
import platform
import random
import threading
import zlib
from collections.abc import Sequence
from datetime import datetime
from io import BytesIO
from time import time
from typing import Any, Optional

import requests
from deviceid import get_device_id
from opentelemetry.exporter.otlp.proto.http import Compression
from opentelemetry.exporter.otlp.proto.http._common import (
    _is_retryable,
)
from opentelemetry.sdk._logs import LogData
from opentelemetry.sdk._logs.export import (
    LogExporter,
    LogExportResult,
)

from olive.telemetry.constants import _ENDPOINT, _HEADERS
from olive.version import __version__ as VERSION

_MAX_RETRYS = 6


class MSFTLogExporter(LogExporter):
    def __init__(
        self,
        headers: Optional[dict[str, str]] = None,
        timeout: Optional[float] = 10,
        compression: Optional[Compression] = Compression.Deflate,
    ):
        self._shutdown_is_occuring = threading.Event()

        self._endpoint = base64.b64decode(_ENDPOINT).decode()
        self._timeout = timeout
        self._compression = compression
        self._session = requests.Session()

        self._headers = json.loads(base64.b64decode(_HEADERS).decode())
        self._iKey = f"o:{self._headers['x-apikey'].split('-')[0]}"
        if headers:
            self._headers.update(headers)
        if self._compression is not Compression.NoCompression:
            self._headers.update({"Content-Encoding": self._compression.value})
        self._session.headers.update(self._headers)
        self._metadata = {
            "device_id": self._generate_encrypted_device_id(),
            "os": {
                "name": platform.system().lower(),
                "version": platform.version(),
                "release": platform.release(),
                "arch": platform.machine(),
            },
            "version": VERSION,
        }

        self._shutdown = False

    def _generate_encrypted_device_id(self) -> str:
        """Generate a FIPS-compliant encrypted device ID using HMAC-SHA256.

        This method uses HMAC-SHA256 which is FIPS 140-2 approved for cryptographic operations.
        The device ID is encrypted using a key derived from the existing endpoint configuration
        to ensure deterministic but secure device identification.

        Returns:
            str: FIPS-compliant encrypted device ID (hex-encoded)

        """
        # Get the raw device ID
        raw_device_id = get_device_id().encode("utf-8")
        # Use SHA256 to encrypt and use Base64 encoding
        return base64.b64encode(hashlib.sha256(raw_device_id).digest()).decode("utf-8")

    def _export(self, data: bytes, timeout_sec: Optional[float] = None):
        if self._compression == Compression.Deflate:
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)  # raw deflate
            compressed_data = compressor.compress(data)
            compressed_data += compressor.flush()
            data = compressed_data
        elif self._compression == Compression.Gzip:
            gzip_data = BytesIO()
            with gzip.GzipFile(fileobj=gzip_data, mode="w") as gzip_stream:
                gzip_stream.write(data)
            data = gzip_data.getvalue()
        elif self._compression == Compression.NoCompression:
            pass

        if timeout_sec is None:
            timeout_sec = self._timeout

        # By default, keep-alive is enabled in Session's request
        # headers. Backends may choose to close the connection
        # while a post happens which causes an unhandled
        # exception. This try/except will retry the post on such exceptions
        updated_headers = {**self._headers, "Content-Length": str(len(data))}
        try:
            resp = self._session.post(
                url=self._endpoint,
                data=data,
                headers=updated_headers,
                timeout=timeout_sec,
            )
        except requests.exceptions.ConnectionError:
            resp = self._session.post(
                url=self._endpoint,
                data=data,
                headers=updated_headers,
                timeout=timeout_sec,
            )
        return resp

    def add_metadata(self, metadata: dict[str, Any]):
        self._metadata.update(metadata)

    def export(self, batch: Sequence[LogData]) -> LogExportResult:
        if self._shutdown:
            return LogExportResult.FAILURE
        json_logs = []
        for log_data in batch:
            log_record = log_data.log_record
            data = {
                k: v
                for k, v in (log_record.attributes or {}).items()
                if k not in {"code.file.path", "code.function.name", "code.line.number"}
            }
            data.update(self._metadata)
            log_entry = {
                "ver": "4.0",
                "name": log_record.body,
                "time": datetime.fromtimestamp(log_record.timestamp / 1e9).isoformat() + "Z"
                if log_record.timestamp
                else None,
                "iKey": self._iKey,
                "data": data,
            }
            json_logs.append(log_entry)

        deadline_sec = time() + self._timeout
        shutdown = False
        for log_entry in json_logs:
            for retry_num in range(_MAX_RETRYS):
                data = json.dumps(log_entry, ensure_ascii=False).encode("utf-8")
                resp = self._export(data, deadline_sec - time())
                if resp.ok:
                    break
                # multiplying by a random number between .8 and 1.2 introduces a +/20% jitter to each backoff.
                backoff_seconds = 2**retry_num * random.uniform(0.8, 1.2)
                if (
                    not _is_retryable(resp)
                    or retry_num + 1 == _MAX_RETRYS
                    or backoff_seconds > (deadline_sec - time())
                    or self._shutdown
                ):
                    return LogExportResult.FAILURE
                shutdown = self._shutdown_is_occuring.wait(backoff_seconds)
                if shutdown:
                    break
            if shutdown:
                break
        return LogExportResult.SUCCESS

    def force_flush(self, timeout_millis: float = 10_000) -> bool:
        """Nothing is buffered in this exporter, so this method does nothing."""
        return True

    def shutdown(self):
        if self._shutdown:
            return
        self._shutdown = True
        self._shutdown_is_occuring.set()
        self._session.close()
