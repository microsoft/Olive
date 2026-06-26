# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import os
import platform
import tempfile
from pathlib import Path

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"


def _resolve_home_dir() -> Path:
    """Resolve the user home directory with fallbacks for container environments."""
    home = os.getenv("HOME")
    if home:
        return Path(home).expanduser()
    try:
        return Path.home()
    except (RuntimeError, KeyError):
        # /var/tmp persists across reboots unlike /tmp (FHS spec)
        if platform.system() != "Windows":
            return Path("/var/tmp")
        return Path(tempfile.gettempdir())


@functools.lru_cache(maxsize=1)
def get_telemetry_base_dir() -> Path:
    os_name = platform.system()
    if os_name == "Windows":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if not base_dir:
            base_dir = str(Path.home() / "AppData" / "Local")
        return Path(base_dir) / "Microsoft" / ".onnxruntime"

    if os_name == "Darwin":
        home = _resolve_home_dir()
        return home / "Library" / "Application Support" / ORT_SUPPORT_DIR

    # Use XDG_CACHE_HOME if set, otherwise fall back to $HOME/.cache
    cache_dir = os.getenv("XDG_CACHE_HOME")
    if not cache_dir:
        cache_dir = str(_resolve_home_dir() / ".cache")

    return Path(cache_dir).expanduser() / ORT_SUPPORT_DIR
