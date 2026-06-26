# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""HTTP transport for the OneCollector exporter (standard library only).

Posts Common Schema JSON to the OneCollector endpoint using ``urllib`` so the
telemetry pipeline has no third-party dependency.
"""

import gzip
import urllib.error
import urllib.request
import zlib
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Optional

from .event_source import event_source
from .options import CompressionType

if TYPE_CHECKING:
    from .callback_manager import CallbackManager, PayloadTransmittedCallbackArgs


class ITransport(ABC):
    """Abstract base class for transports."""

    @abstractmethod
    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        """Send a payload. Returns (success, status_code)."""

    @abstractmethod
    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        """Register a callback for payload transmission events."""


class HttpJsonPostTransport(ITransport):
    """HTTP JSON POST transport using ``urllib`` (no third-party dependency)."""

    def __init__(
        self,
        endpoint: str,
        ikey: str,
        compression: CompressionType,
        callback_manager: Optional["CallbackManager"] = None,
        sdk_version: str = "py-genai-1.0.0",
    ):
        self.endpoint = endpoint
        self.ikey = ikey
        self.compression = compression
        self.sdk_version = sdk_version
        self.callback_manager = callback_manager

        self.headers = {
            "x-apikey": ikey,
            "User-Agent": "Python/3 urllib",
            "Content-Type": "application/x-json-stream; charset=utf-8",
            "sdk-version": sdk_version,
            "NoResponseBody": "true",
        }
        if compression != CompressionType.NO_COMPRESSION:
            self.headers["Content-Encoding"] = compression.value

    def register_payload_transmitted_callback(
        self, callback: Callable[["PayloadTransmittedCallbackArgs"], None], include_failures: bool = False
    ) -> Callable[[], None]:
        if self.callback_manager is None:
            from .callback_manager import CallbackManager

            self.callback_manager = CallbackManager()

        return self.callback_manager.register(callback, include_failures)

    def send(self, payload: bytes, timeout_sec: float, item_count: int = 1) -> tuple[bool, Optional[int]]:
        """Send payload via HTTP POST. Returns (success, status_code)."""
        payload_size_bytes = len(payload)
        try:
            compressed_payload = self._compress(payload)
            headers = {**self.headers, "Content-Length": str(len(compressed_payload))}
            request = urllib.request.Request(
                url=self.endpoint, data=compressed_payload, headers=headers, method="POST"
            )

            success, status_code = self._do_request(request, timeout_sec)

            self._notify(success, status_code, payload_size_bytes, item_count, payload)

            if success:
                return True, status_code
            if event_source.is_error_logging_enabled and status_code is not None:
                event_source.http_transport_error_response("HttpJsonPost", status_code, "", "")
            return False, status_code

        except Exception as ex:
            self._notify(False, None, payload_size_bytes, item_count, payload)
            event_source.transport_exception_thrown("HttpJsonPost", ex)
            return False, None

    @staticmethod
    def _do_request(request: "urllib.request.Request", timeout_sec: float) -> tuple[bool, Optional[int]]:
        """Perform the request, retrying once on a transient connection error."""
        for attempt in range(2):
            try:
                with urllib.request.urlopen(request, timeout=timeout_sec) as response:
                    response.read()
                    status = getattr(response, "status", response.getcode())
                    return (200 <= status < 300, status)
            except urllib.error.HTTPError as http_err:
                # Server responded with a non-2xx status (4xx/5xx): not retried here.
                try:
                    http_err.read()
                except Exception:
                    pass
                return (False, http_err.code)
            except (urllib.error.URLError, TimeoutError, OSError):
                # Connection-level failure: retry once, then give up.
                if attempt == 0:
                    continue
                return (False, None)
        return (False, None)

    def _notify(
        self, success: bool, status_code: Optional[int], payload_size_bytes: int, item_count: int, payload: bytes
    ) -> None:
        if not self.callback_manager:
            return
        from .callback_manager import PayloadTransmittedCallbackArgs

        self.callback_manager.notify(
            PayloadTransmittedCallbackArgs(
                succeeded=success,
                status_code=status_code,
                payload_size_bytes=payload_size_bytes,
                item_count=item_count,
                payload_bytes=payload,
            )
        )

    def _compress(self, data: bytes) -> bytes:
        if self.compression == CompressionType.DEFLATE:
            compressor = zlib.compressobj(wbits=-zlib.MAX_WBITS)
            return compressor.compress(data) + compressor.flush()
        elif self.compression == CompressionType.GZIP:
            gzip_buffer = BytesIO()
            with gzip.GzipFile(fileobj=gzip_buffer, mode="w") as gzip_file:
                gzip_file.write(data)
            return gzip_buffer.getvalue()
        return data

    @staticmethod
    def is_retryable(status_code: Optional[int]) -> bool:
        """Whether a response status indicates the request should be retried."""
        if status_code is None:
            return True  # Network errors are retryable
        return status_code in {408, 429, 500, 502, 503, 504}
