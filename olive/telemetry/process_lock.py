# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""Cross-platform single-holder advisory lock (standard library only).

Used so that, when several processes on one device share a telemetry database,
only one of them runs the uploader's drain loop at a time. Other processes keep
writing events durably to the store; the lock holder drains everyone's rows.
This avoids the same event being uploaded twice by concurrent drainers without
needing per-row reservation bookkeeping.

The lock is an OS advisory lock on a sidecar file (``msvcrt`` on Windows,
``fcntl`` on POSIX). It is released explicitly and also by the OS when the
process exits, so a crashed holder never blocks other processes permanently.
"""

import errno
import os
import time
from contextlib import suppress


class ProcessDrainLock:
    """Non-blocking exclusive advisory lock backed by a sidecar file."""

    def __init__(self, lock_path: str):
        self._lock_path = lock_path
        self._fh = None
        self._posix_lock_api = None

    @property
    def held(self) -> bool:
        return self._fh is not None

    def acquire(self, timeout_seconds: float = 0.0) -> bool:
        """Try to acquire the lock without blocking. Returns True if held."""
        if self._fh is not None:
            return True
        deadline = time.monotonic() + max(0.0, timeout_seconds)
        while True:
            fh = None
            try:
                with suppress(Exception):
                    os.makedirs(os.path.dirname(self._lock_path), exist_ok=True)
                # The handle must remain open while the advisory lock is held.
                fh = open(self._lock_path, "a+b")  # noqa: SIM115  # pylint: disable=consider-using-with
                if os.name == "nt":
                    import msvcrt

                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl

                    try:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self._posix_lock_api = "flock"
                    except AttributeError:
                        fcntl.lockf(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self._posix_lock_api = "lockf"
                    except OSError as exc:
                        unsupported_errors = {errno.ENOSYS}
                        if hasattr(errno, "ENOTSUP"):
                            unsupported_errors.add(errno.ENOTSUP)
                        if hasattr(errno, "EOPNOTSUPP"):
                            unsupported_errors.add(errno.EOPNOTSUPP)
                        if exc.errno not in unsupported_errors:
                            raise
                        fcntl.lockf(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        self._posix_lock_api = "lockf"
                self._fh = fh
                return True
            except Exception:
                self._posix_lock_api = None
                if fh is not None:
                    with suppress(Exception):
                        fh.close()
                if time.monotonic() >= deadline:
                    return False
                time.sleep(min(0.01, max(0.0, deadline - time.monotonic())))

    def release(self) -> None:
        if self._fh is None:
            return
        fh = self._fh
        posix_lock_api = self._posix_lock_api
        self._fh = None
        self._posix_lock_api = None
        try:
            if os.name == "nt":
                import msvcrt

                try:
                    fh.seek(0)
                    msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, 1)
                except Exception:
                    # The OS releases the lock when the handle closes below.
                    pass
            else:
                import fcntl

                try:
                    if posix_lock_api == "lockf":
                        fcntl.lockf(fh.fileno(), fcntl.LOCK_UN)
                    else:
                        fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
                except Exception:
                    # The OS releases the lock when the handle closes below.
                    pass
        finally:
            try:
                fh.close()
            except Exception:
                # Lock cleanup must never fail telemetry callers.
                pass
