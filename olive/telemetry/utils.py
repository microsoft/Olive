# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import functools
import os
import platform
from pathlib import Path

ORT_SUPPORT_DIR = r"Microsoft/DeveloperTools/.onnxruntime"


def _resolve_home_dir() -> Path:
    """Resolve the user home directory with fallbacks for container environments."""
    home = os.getenv("HOME")
    if home and Path(home).is_absolute():
        return Path(home)
    if platform.system() != "Windows":
        try:
            import pwd

            passwd_home = Path(pwd.getpwuid(os.getuid()).pw_dir)
            if passwd_home.is_absolute():
                return passwd_home
        except (AttributeError, ImportError, KeyError, OSError):
            pass
    try:
        fallback_home = Path.home()
        if fallback_home.is_absolute():
            return fallback_home
    except (RuntimeError, KeyError):
        pass
    raise RuntimeError("No absolute per-user telemetry storage directory is available")


@functools.lru_cache(maxsize=1)
def get_telemetry_base_dir() -> Path:
    os_name = platform.system()
    if os_name == "Windows":
        base_dir = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA")
        if base_dir and Path(base_dir).is_absolute():
            return Path(base_dir) / ORT_SUPPORT_DIR
        return _resolve_home_dir() / "AppData" / "Local" / ORT_SUPPORT_DIR

    if os_name == "Darwin":
        home = _resolve_home_dir()
        return home / "Library" / "Application Support" / ORT_SUPPORT_DIR

    # Use XDG_CACHE_HOME if set, otherwise fall back to $HOME/.cache
    cache_dir = os.getenv("XDG_CACHE_HOME")
    if cache_dir and Path(cache_dir).is_absolute():
        return Path(cache_dir) / ORT_SUPPORT_DIR
    return _resolve_home_dir() / ".cache" / ORT_SUPPORT_DIR
