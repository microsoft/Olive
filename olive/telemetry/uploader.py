# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Background uploader that drains the SQLite offline store to OneCollector.

Reads the oldest batch of events, POSTs them, and:
- deletes them on success (HTTP 2xx),
- deletes them on a permanent, non-retryable failure (e.g. HTTP 4xx) so a
  poison event cannot block the queue forever,
- leaves them on a transient failure (network error / HTTP 5xx / timeout) to be
  retried on the next cycle or the next process run.

Durability is provided by the on-disk store, so the process can exit at any time
without losing events and without an exit-time flush.
"""

import threading
import time
from typing import Optional

from olive.telemetry.library.options import CompressionType, OneCollectorTransportOptions
from olive.telemetry.library.payload_builder import PayloadBuilder
from olive.telemetry.library.transport import HttpJsonPostTransport
from olive.telemetry.offline_store import OfflineEventStore
from olive.telemetry.process_lock import ProcessDrainLock


class EventUploader:
    """Drains the offline store and ships events over HTTP on a daemon thread.

    When several processes share one telemetry database, a single-holder
    advisory lock ensures only one of them drains at a time, so the same event
    is never uploaded by two concurrent drainers. Processes that do not hold the
    lock keep writing events durably; the holder drains them.
    """

    def __init__(
        self,
        store: OfflineEventStore,
        instrumentation_key: str,
        endpoint: str = OneCollectorTransportOptions.DEFAULT_ENDPOINT,
        compression: CompressionType = CompressionType.DEFLATE,
        drain_interval_seconds: float = 2.0,
        max_items_per_drain: int = 256,
        send_timeout_seconds: float = 10.0,
        idle_backoff_seconds: float = 30.0,
    ):
        self._store = store
        self._drain_interval = drain_interval_seconds
        self._max_items = max_items_per_drain
        self._send_timeout = send_timeout_seconds
        self._idle_backoff = idle_backoff_seconds
        self._drain_lock = ProcessDrainLock(store.db_path + ".lock")

        self._transport = HttpJsonPostTransport(
            endpoint=endpoint,
            ikey=instrumentation_key,
            compression=compression,
        )

        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None

    # ----- control -------------------------------------------------------

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="genai-telemetry-uploader", daemon=True)
        self._thread.start()

    def request_drain(self) -> None:
        """Nudge the uploader to drain promptly (e.g. after logging an event)."""
        self._wake.set()

    def stop_loop(self, join_timeout_seconds: float = 5.0) -> bool:
        """Stop the background loop and wait briefly for it to exit.

        Returns True if the thread actually stopped (so a caller may safely drain
        as the sole drainer), False if it is still alive (e.g. stuck in an
        in-flight send) — in which case the caller must NOT drain, to avoid
        double-sending the rows the thread is still processing.
        """
        self._stop.set()
        self._wake.set()
        thread = self._thread
        if thread is None:
            return True
        thread.join(join_timeout_seconds)
        stopped = not thread.is_alive()
        if stopped:
            self._thread = None
        return stopped

    def signal_stop(self) -> None:
        """Ask the loop to stop without blocking the caller.

        The daemon thread winds down on its next wake; the drain lock is released
        when it exits (or by the OS at process exit). Use this for the opt-out
        path so disabling telemetry never blocks the host.
        """
        self._stop.set()
        self._wake.set()

    def close(self) -> None:
        """Release the single-drainer lock (urllib holds no persistent connection)."""
        self._drain_lock.release()

    def stop(self, timeout_seconds: float = 12.0) -> None:
        """Stop the loop and release the drain lock (convenience)."""
        self.stop_loop(timeout_seconds)
        self.close()

    # ----- draining ------------------------------------------------------

    def drain_once(self) -> tuple[int, int]:
        """Attempt to upload one batch. Returns (delivered_count, left_count).

        ``left_count`` is non-zero only when a transient failure leaves rows on
        disk for a later retry; permanently-rejected rows are dropped (counted as
        delivered for loop-termination purposes since they leave the queue).
        """
        batch = self._store.get_batch(self._max_items)
        if not batch:
            return (0, 0)

        builder = PayloadBuilder(
            max_size_bytes=OneCollectorTransportOptions.DEFAULT_MAX_PAYLOAD_SIZE_BYTES,
            max_items=OneCollectorTransportOptions.DEFAULT_MAX_ITEMS_PER_PAYLOAD,
        )
        included: list[int] = []
        for row_id, payload in batch:
            if not builder.can_add(payload) and not builder.is_empty:
                break
            builder.add(payload)
            included.append(row_id)
        payload_bytes = builder.build()

        try:
            success, status = self._transport.send(payload_bytes, self._send_timeout, item_count=len(included))
        except Exception:
            success, status = (False, None)

        if success:
            self._store.delete(included)
            return (len(included), 0)
        if not HttpJsonPostTransport.is_retryable(status):
            # Permanent rejection (e.g. 4xx): drop so it can't block the queue.
            self._store.delete(included)
            return (len(included), 0)
        # Transient failure: leave the rows for the next attempt.
        return (0, len(included))

    def flush(self, max_seconds: float = 5.0) -> None:
        """Best-effort drain of all pending events, bounded by max_seconds.

        Only drains if this process holds the single-drainer lock; otherwise the
        events stay durably on disk for the lock holder (or the next run).
        """
        if not self._drain_lock.acquire():
            return
        deadline = time.time() + max_seconds
        while time.time() < deadline:
            delivered, left = self.drain_once()
            if delivered == 0 and left == 0:
                return  # queue empty
            if left:
                return  # transient failure; leave the rest for next run

    def _run(self) -> None:
        try:
            while not self._stop.is_set():
                transient_failure = 0
                # Only one process drains at a time. If another holds the lock, skip
                # draining this cycle; our events remain durable for the holder.
                if self._drain_lock.acquire():
                    try:
                        delivered, left = self.drain_once()
                        while delivered > 0 and not self._stop.is_set():
                            delivered, left = self.drain_once()
                        transient_failure = left
                    except Exception:
                        transient_failure = 1

                wait = self._idle_backoff if transient_failure else self._drain_interval
                self._wake.wait(wait)
                self._wake.clear()
        finally:
            # Release the single-drainer lock when the loop exits so another
            # process can take over (also released by close()/OS on exit).
            self._drain_lock.release()
