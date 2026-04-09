# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import asyncio
import atexit
import json
import os
import re
import sys
import uuid
from datetime import datetime

from olive_mcp.constants import (
    OUTPUT_BASE,
    SUPPORTED_QUANT_ALGORITHMS,
    WORKER_PATH,
    Command,
    SupportedPrecision,
    SupportedProvider,
)
from olive_mcp.packages import _resolve_packages
from olive_mcp.venv import _get_or_create_venv

# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}

_MAX_CONCURRENT_JOBS = 3

_JOB_TTL_SECONDS = 3600  # purge finished jobs after 1 hour


def _cleanup_jobs():
    """Terminate all running background processes on server shutdown."""
    for job in _jobs.values():
        proc = job.get("process")
        if proc and proc.returncode is None:
            try:
                if sys.platform == "win32":
                    import subprocess

                    subprocess.call(
                        ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                        stdout=subprocess.DEVNULL,
                        stderr=subprocess.DEVNULL,
                    )
                else:
                    proc.terminate()
            except Exception:
                # Best-effort cleanup at shutdown — nothing to do if termination fails
                continue


atexit.register(_cleanup_jobs)


def _purge_old_jobs():
    """Remove completed/errored jobs older than _JOB_TTL_SECONDS."""
    now = datetime.now()
    to_delete = [
        jid
        for jid, job in _jobs.items()
        if job["status"] in ("completed", "error")
        and (now - datetime.fromisoformat(job.get("last_activity", job["started_at"]))).total_seconds()
        > _JOB_TTL_SECONDS
    ]
    for jid in to_delete:
        del _jobs[jid]


def _create_job(command: str, description: str) -> str:
    """Create a new background job and return its ID."""
    _purge_old_jobs()
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "starting",
        "command": command,
        "description": description,
        "log_lines": [],
        "result": None,
        "process": None,
        "started_at": datetime.now().isoformat(),
    }
    return job_id


def _job_log(job_id: str, line: str):
    """Append a log line to a job."""
    if job_id in _jobs:
        _jobs[job_id]["log_lines"].append(line)
        _jobs[job_id]["last_activity"] = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_output_path(prefix: str, model_name: str) -> str:
    safe_name = model_name.replace("/", "_").replace("\\", "_")
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = OUTPUT_BASE / f"{prefix}_{safe_name}_{ts}"
    path.mkdir(parents=True, exist_ok=True)
    return str(path)


def _build_kwargs(**kw) -> dict:
    """Filter out None values from kwargs."""
    return {k: v for k, v in kw.items() if v is not None}


# ---------------------------------------------------------------------------
# Worker execution (background)
# ---------------------------------------------------------------------------


async def _run_olive_background(
    job_id: str,
    command: str,
    kwargs: dict,
    packages: list[str],
    hf_token: str | None = None,
):
    """Background task: run olive in isolated venv, stream logs to job."""
    try:
        _jobs[job_id]["status"] = "setting_up"
        _job_log(job_id, f"Packages to install: {', '.join(packages)}")
        python_path = await _get_or_create_venv(packages, job_id, _job_log)

        _jobs[job_id]["status"] = "running"
        _job_log(job_id, f"Running: olive {command}")

        # Pass HF token via environment variable (not in kwargs — olive API doesn't take it)
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
            _job_log(job_id, "HuggingFace token provided")

        proc = await asyncio.create_subprocess_exec(
            str(python_path),
            "-u",
            str(WORKER_PATH),
            command,
            json.dumps(kwargs),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
            limit=10 * 1024 * 1024,  # 10 MB line limit (default 64KB is too small for olive output)
        )
        _jobs[job_id]["process"] = proc

        # Stream ALL stderr (olive logs) directly into job log
        while True:
            try:
                line = await proc.stderr.readline()
            except ValueError:
                # Line exceeded even the 10MB limit — skip it
                continue
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                # Truncate extremely long lines for display (e.g. base64 blobs)
                if len(decoded) > 500:
                    decoded = decoded[:500] + "... (truncated)"
                _job_log(job_id, decoded)

        # Read stdout (JSON result)
        stdout_bytes = await proc.stdout.read()
        await proc.wait()
        stdout_str = stdout_bytes.decode("utf-8", errors="replace")

        # If the job was already cancelled, don't overwrite its status.
        if _jobs[job_id]["status"] == "cancelled":
            return

        if proc.returncode != 0:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {
                "status": "error",
                "returncode": proc.returncode,
                "error": stdout_str[-3000:],
            }
            _job_log(job_id, f"Failed with exit code {proc.returncode}")
        else:
            try:
                result = json.loads(stdout_str)
            except json.JSONDecodeError:
                result = {"status": "error", "error": "Failed to parse output", "raw": stdout_str[-3000:]}

            _jobs[job_id]["status"] = "completed" if result.get("status") == "success" else "error"
            _jobs[job_id]["result"] = result
            _job_log(
                job_id, "Completed successfully" if _jobs[job_id]["status"] == "completed" else "Completed with errors"
            )

        # Attach smart error suggestions for failed jobs
        if _jobs[job_id]["status"] == "error":
            error_text = _jobs[job_id]["result"].get("error", "")
            error_text += "\n" + "\n".join(_jobs[job_id]["log_lines"][-30:])
            suggestions = _get_error_suggestions(error_text, command, kwargs)
            if suggestions:
                _jobs[job_id]["result"]["suggestions"] = suggestions

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["result"] = {"status": "error", "error": str(exc)}
        error_text = str(exc) + "\n" + "\n".join(_jobs[job_id].get("log_lines", [])[-30:])
        suggestions = _get_error_suggestions(error_text, command, kwargs)
        if suggestions:
            _jobs[job_id]["result"]["suggestions"] = suggestions
        _job_log(job_id, f"Exception: {exc}")


_INCOMPATIBLE_PRECISION = {
    "CPUExecutionProvider": {"fp16", "bf16"},
}


# ---------------------------------------------------------------------------
# Phase detection — structured progress from raw Olive logs
# ---------------------------------------------------------------------------

_PHASE_PATTERNS: list[tuple[re.Pattern, str, str]] = [
    (re.compile(r"Saving|saving|output model|Output model", re.I), "saving", "Saving output model"),
    (re.compile(r"Evaluating|evaluation|Running evaluation|lm_eval", re.I), "evaluating", "Evaluating model quality"),
    (re.compile(r"Quantiz", re.I), "quantizing", "Quantizing model weights"),
    (re.compile(r"Training|training|fine.?tun|Epoch \[", re.I), "training", "Training / fine-tuning"),
    (re.compile(r"Running pass|Pass \[", re.I), "running_pass", "Running optimization pass"),
    (re.compile(r"Conversion|Converting|exporting|capture", re.I), "converting", "Converting model format"),
    (
        re.compile(r"Loading model|loading model|from_pretrained|Loading checkpoint", re.I),
        "loading_model",
        "Loading model into memory",
    ),
    (re.compile(r"[Dd]ownloading|Fetching.*model", re.I), "downloading", "Downloading model files"),
    (re.compile(r"Creating venv|Installing packages|Venv ready", re.I), "setting_up_env", "Setting up environment"),
]

_PASS_NAME_RE = re.compile(r"Pass \[(\w+)\]|Running pass (\w+)", re.I)


def _detect_phase(log_lines: list[str]) -> dict:
    """Detect current execution phase from recent log lines."""
    result: dict = {"phase": "processing", "phase_description": "Processing"}
    for line in reversed(log_lines[-20:]):
        for pattern, phase_key, description in _PHASE_PATTERNS:
            if pattern.search(line):
                result["phase"] = phase_key
                result["phase_description"] = description
                m = _PASS_NAME_RE.search(line)
                if m:
                    result["current_pass"] = m.group(1) or m.group(2)
                return result
    return result


# ---------------------------------------------------------------------------
# Smart error suggestions
# ---------------------------------------------------------------------------

_ERROR_PATTERNS: list[tuple[re.Pattern, str]] = [
    (
        re.compile(r"CUDA out of memory|OOM|OutOfMemoryError|torch\.OutOfMemoryError", re.I),
        "Out of GPU memory. Try: (1) Use int4 precision instead of fp16, "
        "(2) Use quantize with algorithm='rtn' instead of optimize, (3) Use a smaller model, "
        "(4) Close other GPU applications.",
    ),
    (
        re.compile(
            r"cuda.*not available|No CUDA GPUs|CUDAExecutionProvider.*not available|CUDA driver|CUDA error|cudnn", re.I
        ),
        "CUDA is not available. Use provider='CPUExecutionProvider' or "
        "provider='DmlExecutionProvider' (Windows with DirectX 12 GPU).",
    ),
    (
        re.compile(r"401|403|authentication|gated|Access denied|forbidden", re.I),
        "Authentication required. This model may be gated on HuggingFace. "
        "Get a token at https://huggingface.co/settings/tokens and retry with hf_token.",
    ),
    (
        re.compile(r"404|Repository Not Found|does not exist on the Hub", re.I),
        "Model not found. Check the model name is spelled correctly and exists on HuggingFace. "
        "For private models, provide hf_token.",
    ),
    (
        re.compile(r"No module named|ModuleNotFoundError|ImportError", re.I),
        "Missing Python dependency. The cached venv may be stale. "
        "Try deleting ~/.olive-mcp/venvs and retrying, or use a different algorithm/implementation.",
    ),
    (
        re.compile(r"fp16.*CPU|float16.*cpu|CPU.*does not support.*fp16|half precision.*cpu", re.I),
        "CPU does not support fp16 precision. Use int4, int8, or fp32 instead.",
    ),
    (
        re.compile(r"No space left|OSError.*Errno 28|disk.?space|DiskFull", re.I),
        "Disk full. Free up space or use manage_outputs(action='delete') to remove old results.",
    ),
    (
        re.compile(r"timed?\s*out|TimeoutError", re.I),
        "Operation timed out. The model may be too large. Try a smaller model or increase resources.",
    ),
    (
        re.compile(r"onnxruntime.*version|ort.*version|incompatible.*version", re.I),
        "OnnxRuntime version conflict. Delete ~/.olive-mcp/venvs to force a fresh environment.",
    ),
    (
        re.compile(r"ConnectionError|SSLError|network|NameResolutionError|MaxRetryError", re.I),
        "Network error. Check your internet connection. If behind a proxy, set "
        "HTTP_PROXY/HTTPS_PROXY environment variables.",
    ),
]


def _get_error_suggestions(error_str: str, command: str, kwargs: dict) -> list[str]:
    """Match error text against known patterns and return actionable suggestions."""
    suggestions = []
    seen = set()
    for pattern, suggestion in _ERROR_PATTERNS:
        if pattern.search(error_str) and suggestion not in seen:
            suggestions.append(suggestion)
            seen.add(suggestion)

    # Command-specific fallback
    if not suggestions:
        if command == Command.OPTIMIZE and kwargs.get("precision") in ("int4", "uint4"):
            suggestions.append(
                "int4 optimization with GPTQ can be very slow on CPU (30min+). "
                "For faster int4, use quantize(algorithm='rtn') instead."
            )
        if command == Command.FINETUNE and kwargs.get("method") == "lora":
            suggestions.append(
                "If running out of memory during fine-tuning, try method='qlora' "
                "which uses 4-bit quantization to reduce memory usage."
            )

    return suggestions


def _validate_params(command: str, kwargs: dict) -> str | None:
    """Validate parameters before starting a job. Returns error message or None."""
    provider = kwargs.get("provider")
    precision = kwargs.get("precision")

    # Check provider is recognized
    if provider and provider not in SupportedProvider:
        return f"Unknown provider '{provider}'. Supported: {', '.join(SupportedProvider)}"

    # Check precision is recognized
    if precision and precision not in SupportedPrecision:
        return f"Unknown precision '{precision}'. Supported: {', '.join(SupportedPrecision)}"

    # Check provider-precision compatibility
    if provider and precision:
        blocked = _INCOMPATIBLE_PRECISION.get(provider, set())
        if precision in blocked:
            return f"{provider} does not support {precision} precision. Use int4, int8, or fp32 instead."

    # Check quantization algorithm
    algorithm = kwargs.get("algorithm")
    if algorithm and algorithm not in SUPPORTED_QUANT_ALGORITHMS:
        return f"Unknown algorithm '{algorithm}'. Supported: {', '.join(SUPPORTED_QUANT_ALGORITHMS)}"

    # Check finetune method
    if command == Command.FINETUNE:
        method = kwargs.get("method")
        if method and method not in ("lora", "qlora"):
            return f"Unknown finetune method '{method}'. Supported: lora, qlora"

    return None


async def _start_job(command: str, description: str, kwargs: dict, hf_token: str | None = None) -> dict:
    """Create a job, resolve packages, launch background task, return immediately."""
    # Validate before creating venv (saves minutes of wasted setup time)
    error = _validate_params(command, kwargs)
    if error:
        return {"status": "error", "error": error}

    # Limit concurrent running jobs to prevent overwhelming the system
    active = sum(1 for j in _jobs.values() if j["status"] in ("starting", "setting_up", "running"))
    if active >= _MAX_CONCURRENT_JOBS:
        return {
            "status": "error",
            "error": f"Too many concurrent jobs ({active} running). Wait for a job to finish or cancel one.",
        }

    job_id = _create_job(command, description)
    packages = _resolve_packages(command, **kwargs)
    task = asyncio.create_task(_run_olive_background(job_id, command, kwargs, packages, hf_token=hf_token))
    _jobs[job_id]["_task"] = task
    return {
        "job_id": job_id,
        "status": "running",
        "packages": packages,
        "message": f"Job started: {description}. Use get_job_status('{job_id}') to check progress.",
    }
