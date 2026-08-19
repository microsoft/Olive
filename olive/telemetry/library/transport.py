# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""HTTP transport for the OneCollector exporter (standard library only).

Posts Common Schema JSON to the OneCollector endpoint using ``urllib`` so the
telemetry pipeline has no third-party dependency.
"""

import gzip
import queue
import threading
import time
import urllib.error
import urllib.request
import zlib
from io import BytesIO
from typing import Callable, Optional

from olive.telemetry.library.options import CompressionType


class HttpJsonPostTransport:
    """HTTP JSON POST transport using ``urllib`` (no third-party dependency)."""

    def __init__(
        self,
        endpoint: str,
        ikey: str,
        compression: CompressionType,
        sdk_version: str = "py-olive-1.0.0",
    ):
        self.endpoint = endpoint
        self.ikey = ikey
        self.compression = compression
        self.sdk_version = sdk_version

        self.headers = {
            "x-apikey": ikey,
            "User-Agent": "Python/3 urllib",
            "Content-Type": "application/x-json-stream; charset=utf-8",
            "sdk-version": sdk_version,
            "NoResponseBody": "true",
        }
        if compression != CompressionType.NO_COMPRESSION:
            self.headers["Content-Encoding"] = compression.value
        self._worker_lock = threading.Lock()
        self._inflight_worker: Optional[threading.Thread] = None
        self._inflight_results = None
        self._inflight_request_key = None

    def send(
        self,
        payload: bytes,
        timeout_sec: float,
        item_count: int = 1,
        on_send_admitted: Optional[Callable[[], None]] = None,
    ) -> tuple[bool, Optional[int]]:
        """Send payload via HTTP POST. Returns (success, status_code)."""
        try:
            compressed_payload = self._compress(payload)
            headers = {**self.headers, "Content-Length": str(len(compressed_payload))}
            request = urllib.request.Request(url=self.endpoint, data=compressed_payload, headers=headers, method="POST")

            if on_send_admitted is not None:
                on_send_admitted()
            success, status_code = self._do_request(request, timeout_sec)
            return success, status_code
        except Exception:
            return False, None

    def _do_request(self, request: "urllib.request.Request", timeout_sec: float) -> tuple[bool, Optional[int]]:
        """Run the request behind a wall-clock deadline, including DNS resolution."""
        request_key = (request.full_url, bytes(request.data or b""))
        with self._worker_lock:
            worker = self._inflight_worker
            if worker is not None:
                if worker.is_alive():
                    return (False, None)
                result = self._consume_inflight_result()
                if self._inflight_request_key == request_key:
                    self._clear_inflight()
                    return result
                self._clear_inflight()

            results = queue.Queue(maxsize=1)
            worker = threading.Thread(
                target=self._run_request_worker,
                args=(request, timeout_sec, results),
                name="olive-telemetry-http",
                daemon=True,
            )
            self._inflight_worker = worker
            self._inflight_results = results
            self._inflight_request_key = request_key
            worker.start()

        worker.join(max(0.0, timeout_sec))
        with self._worker_lock:
            if worker.is_alive():
                return (False, None)
            result = self._consume_inflight_result()
            self._clear_inflight()
            return result

    @staticmethod
    def _run_request_worker(request, timeout_sec: float, results) -> None:
        try:
            result = HttpJsonPostTransport._do_request_blocking(request, timeout_sec)
        except Exception:
            result = (False, None)
        try:
            results.put_nowait(result)
        except queue.Full:
            pass

    def _consume_inflight_result(self) -> tuple[bool, Optional[int]]:
        try:
            return self._inflight_results.get_nowait()
        except (AttributeError, queue.Empty):
            return (False, None)

    def _clear_inflight(self) -> None:
        self._inflight_worker = None
        self._inflight_results = None
        self._inflight_request_key = None

    @staticmethod
    def _do_request_blocking(request: "urllib.request.Request", timeout_sec: float) -> tuple[bool, Optional[int]]:
        """Perform the request, retrying once on a transient connection error."""
        deadline = time.monotonic() + max(0.0, timeout_sec)
        for attempt in range(2):
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                return (False, None)
            try:
                with urllib.request.urlopen(request, timeout=remaining) as response:
                    status = getattr(response, "status", response.getcode())
                    return (200 <= status < 300, status)
            except urllib.error.HTTPError as http_err:
                # Server responded with a non-2xx status (4xx/5xx): not retried here.
                try:
                    http_err.close()
                except Exception:
                    # The HTTP status remains authoritative if response cleanup fails.
                    pass
                return (False, http_err.code)
            except (urllib.error.URLError, TimeoutError, OSError):
                # Connection-level failure: retry once, then give up.
                if attempt == 0:
                    continue
                return (False, None)
        return (False, None)

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
        return status_code in {408, 429} or 500 <= status_code <= 599
