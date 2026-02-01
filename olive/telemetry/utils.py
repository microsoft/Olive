# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import hashlib
import traceback
from types import TracebackType
from typing import Optional

from olive.telemetry.deviceid import get_device_id


def _generate_encrypted_device_id() -> str:
    """Generate a FIPS-compliant encrypted device ID using SHA256.

    This method uses SHA256 which is FIPS 140-2 approved for cryptographic operations.
    The device ID is hashed to ensure deterministic but secure device identification.

    Returns:
        str: FIPS-compliant encrypted device ID (base64-encoded)

    """
    hash_bytes = hashlib.sha256(get_device_id().encode("utf-8")).digest()
    return hash_bytes.hex().upper()


def _format_exception_msg(ex: BaseException, tb: Optional[TracebackType] = None) -> str:
    """Format an exception and trim local paths for readability."""
    folder = "Olive"
    file_line = 'File "'
    formatted = traceback.format_exception(type(ex), ex, tb, limit=5)
    lines = []
    for line in formatted:
        line_trunc = line.strip()
        if line_trunc.startswith(file_line) and folder in line_trunc:
            idx = line_trunc.find(folder)
            if idx != -1:
                line_trunc = line_trunc[idx + len(folder) :]
        elif line_trunc.startswith(file_line):
            idx = line_trunc[len(file_line) :].find('"')
            line_trunc = line_trunc[idx + len(file_line) :]
        lines.append(line_trunc)
    return "\n".join(lines)
