# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
import platform
import traceback
from pathlib import Path
from types import TracebackType
from typing import Optional

DEVICEID_LOCATION = r"Microsoft/DeveloperTools/deviceid/.onnxruntime/"


def get_telemetry_base_dir() -> Path:
    os_name = platform.system()
    if os_name == "Windows":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base_dir:
            base_dir = str(Path.home() / "AppData" / "Local")
        return Path(base_dir) / "Microsoft" / ".onnxruntime"

    if os_name == "Darwin":
        home = os.getenv("HOME")
        if home is None:
            raise ValueError("HOME environment variable not set")
        return Path(home) / "Library" / "Application Support" / DEVICEID_LOCATION

    home = os.getenv("XDG_CACHE_HOME", f"{os.getenv('HOME')}/.cache")
    if not home:
        raise ValueError("HOME environment variable not set")

    return Path(home) / DEVICEID_LOCATION


def _format_exception_message(ex: BaseException, tb: Optional[TracebackType] = None) -> str:
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
