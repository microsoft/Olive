# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""SQLite-backed durable queue for telemetry events.

A deliberately small subset of the Microsoft 1DS C++ SDK offline store
(cpp_client_telemetry/lib/offline/OfflineStorage_SQLite.cpp): a single FIFO
table of serialized event payloads. An uploader drains it, deleting rows on
success, dropping them on a permanent (non-retryable) send result, and leaving
them for the next attempt on a transient failure. Because every event is
written to disk before any network call, the process can exit at any time
without losing data and without an exit-time flush.

Uses only the Python standard library (``sqlite3``), so it adds no dependency.

Intentionally omitted from the full 1DS store (not needed for low-volume CLI
telemetry): per-event priority (``latency``), persistence classes,
reservation/leasing (``reserved_until``), per-row retry counters, tenant
multiplexing, and the ``settings`` table. The schema version is tracked with
SQLite's built-in ``PRAGMA user_version``.
"""

import os
import sqlite3
import threading
from typing import Optional

SCHEMA_VERSION = 1


def _chmod_best_effort(path: str, mode: int) -> None:
    if os.name == "nt":
        return
    try:
        os.chmod(path, mode)
    except OSError:
        pass


class OfflineEventStore:
    """Durable FIFO queue of serialized telemetry event payloads.

    All methods are best-effort and swallow storage errors: telemetry must
    never crash the host application. Thread-safe via a per-instance lock;
    tolerant of concurrent processes via WAL mode + ``busy_timeout``.
    """

    def __init__(self, db_path: str, max_records: int = 2048, busy_timeout_ms: int = 3000):
        self._db_path = db_path
        self._max_records = max_records
        # When full, trim back to this watermark so we don't trim on every insert.
        self._trim_target = max(1, (max_records * 3) // 4)
        self._busy_timeout_ms = busy_timeout_ms
        self._lock = threading.Lock()
        self._conn: Optional[sqlite3.Connection] = None
        self._initialize()

    def _initialize(self) -> None:
        parent = os.path.dirname(self._db_path)
        try:
            os.makedirs(parent, mode=0o700, exist_ok=True)
            _chmod_best_effort(parent, 0o700)
        except Exception:
            pass
        try:
            conn = sqlite3.connect(
                self._db_path, timeout=self._busy_timeout_ms / 1000.0, check_same_thread=False
            )
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(f"PRAGMA busy_timeout={self._busy_timeout_ms}")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY AUTOINCREMENT, payload BLOB NOT NULL)"
            )
            if conn.execute("PRAGMA user_version").fetchone()[0] == 0:
                conn.execute(f"PRAGMA user_version={SCHEMA_VERSION}")
            conn.commit()
            self._conn = conn
            self._harden_permissions()
        except Exception:
            self._conn = None

    def _harden_permissions(self) -> None:
        _chmod_best_effort(os.path.dirname(self._db_path), 0o700)
        for path in (self._db_path, self._db_path + "-wal", self._db_path + "-shm"):
            if os.path.exists(path):
                _chmod_best_effort(path, 0o600)

    @property
    def is_open(self) -> bool:
        return self._conn is not None

    @property
    def db_path(self) -> str:
        return self._db_path

    def store(self, payload: bytes) -> bool:
        """Append one serialized event; trims the oldest rows if over capacity."""
        if not payload:
            return False
        with self._lock:
            if self._conn is None:
                return False
            try:
                self._conn.execute("INSERT INTO events (payload) VALUES (?)", (sqlite3.Binary(payload),))
                count = self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0]
                if count > self._max_records:
                    self._conn.execute(
                        "DELETE FROM events WHERE id IN (SELECT id FROM events ORDER BY id ASC LIMIT ?)",
                        (count - self._trim_target,),
                    )
                self._conn.commit()
                self._harden_permissions()
                return True
            except Exception:
                return False

    def get_batch(self, max_count: int) -> list[tuple[int, bytes]]:
        """Return up to ``max_count`` oldest events as (id, payload) pairs."""
        with self._lock:
            if self._conn is None:
                return []
            try:
                rows = self._conn.execute(
                    "SELECT id, payload FROM events ORDER BY id ASC LIMIT ?",
                    (max_count if max_count > 0 else -1,),
                ).fetchall()
                return [(r[0], bytes(r[1])) for r in rows]
            except Exception:
                return []

    def delete(self, ids: list[int]) -> None:
        """Remove rows by id (after a successful upload or a permanent drop)."""
        if not ids:
            return
        with self._lock:
            if self._conn is None:
                return
            try:
                self._conn.executemany("DELETE FROM events WHERE id=?", [(i,) for i in ids])
                self._conn.commit()
            except Exception:
                pass

    def count(self) -> int:
        with self._lock:
            if self._conn is None:
                return 0
            try:
                return int(self._conn.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            except Exception:
                return 0

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                except Exception:
                    pass
                self._conn = None
