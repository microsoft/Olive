# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import asyncio
import hashlib
import shutil
import sys
from datetime import datetime
from pathlib import Path

from olive_mcp.constants import _VENV_MAX_AGE_DAYS, VENV_BASE


def _get_python_path(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _purge_old_venvs():
    """Remove cached venvs not used within _VENV_MAX_AGE_DAYS."""
    if not VENV_BASE.exists():
        return
    now = datetime.now()
    for d in list(VENV_BASE.iterdir()):
        if not d.is_dir():
            continue
        marker = d / ".last_used"
        if marker.exists():
            age = (now - datetime.fromtimestamp(marker.stat().st_mtime)).days
        else:
            # No marker — use directory mtime as fallback
            age = (now - datetime.fromtimestamp(d.stat().st_mtime)).days
        if age > _VENV_MAX_AGE_DAYS:
            shutil.rmtree(d, ignore_errors=True)


def _touch_venv(venv_path: Path):
    """Update the last-used marker for a cached venv."""
    marker = venv_path / ".last_used"
    marker.touch()


async def _get_or_create_venv(packages: list[str], job_id: str, job_log_fn) -> Path:
    """Get or create a cached uv venv with the specified packages.

    Args:
        packages: List of packages to install.
        job_id: Job ID for logging.
        job_log_fn: Callable(job_id, line) to log messages.

    """
    # Periodically purge stale venvs
    _purge_old_venvs()

    key = hashlib.sha256("|".join(sorted(packages)).encode()).hexdigest()[:12]
    venv_path = VENV_BASE / key
    python_path = _get_python_path(venv_path)

    if not python_path.exists():
        job_log_fn(job_id, f"Creating venv with: {', '.join(packages)}")
        VENV_BASE.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "uv",
            "venv",
            str(venv_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {stderr.decode()}")

        job_log_fn(job_id, f"Installing packages: {', '.join(packages)}")
        proc = await asyncio.create_subprocess_exec(
            "uv",
            "pip",
            "install",
            "--python",
            str(python_path),
            *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install packages: {stderr.decode()}")

        job_log_fn(job_id, "Venv ready")
    else:
        job_log_fn(job_id, f"Reusing cached venv ({key})")

    # Log installed packages for debugging
    proc = await asyncio.create_subprocess_exec(
        "uv",
        "pip",
        "list",
        "--python",
        str(python_path),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, _ = await proc.communicate()
    if proc.returncode == 0:
        job_log_fn(job_id, f"Installed packages:\n{stdout.decode().strip()}")

    _touch_venv(venv_path)
    return python_path
