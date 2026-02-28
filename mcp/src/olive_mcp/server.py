"""Olive MCP Server - Model optimization via Microsoft Olive Python API."""

import asyncio
import ctypes
import hashlib
import json
import os
import platform
import shutil
import sys
import uuid
from datetime import datetime
from pathlib import Path

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="olive",
    instructions="""Olive MCP server for Microsoft Olive model optimization.

## How to interact with users

**Be adaptive to the user's expertise level.** Not everyone knows what "int4", "GPTQ", or "execution provider" means.

### If the user gives a specific request (e.g. "quantize Phi-4 to int4 for CPU"):
- They know what they want. Just run it. No extra questions needed.

### If the user gives a vague request (e.g. "optimize this model" or "make it smaller"):
- Do NOT ask multiple technical questions (device, precision, algorithm, etc.).
- Instead, ask ONE simple question with ready-to-go options. Each option should be a
  complete plan described in plain language. Example:

  "How do you want to optimize Phi-4-mini-instruct?"
  1. **Make it as small as possible** — I'll quantize to 4-bit. Best for running on laptops or limited hardware.
  2. **Balance size and quality** — I'll quantize to 8-bit. Good default for most use cases.
  3. **Best quality on GPU** — I'll optimize with fp16. Requires a GPU with 8GB+ VRAM.
  4. **You decide** — Tell me your target device, precision, etc.

  Then run immediately based on their choice. No follow-up questions.

### If the user just says "optimize <model>" with no other context:
- Pick option 2 (balanced/int8) as the default, tell the user what you're doing and why, and run it.
- The user can always ask for something different after seeing the result.

**Key principle: minimize questions, maximize action.** It's better to run with good defaults
and let the user adjust than to interrogate them before starting.

## Device constraints (for YOUR decision-making, not for asking the user)
- **CPU**: Does NOT support fp16. Use int4 or int8. Provider = CPUExecutionProvider.
- **GPU (NVIDIA)**: Supports fp16, int4, int8. Provider = CUDAExecutionProvider.
- **GPU (DirectML/Windows)**: Supports fp16, int4, int8. Provider = DmlExecutionProvider.
- **NPU**: Provider = QNNExecutionProvider. Limited precision support.

If unsure about the user's device, call `detect_hardware` to auto-detect GPU, RAM, and disk space.
Use the result to pick the best provider and precision automatically — no need to ask the user.

## Async job pattern
All long-running tools run in the background and return a `job_id` immediately.
Poll `get_job_status(job_id)` to check progress and get results.

**Workflow:**
1. Call the tool → returns `{"job_id": "xxx", "status": "running"}`
2. Call `get_job_status("xxx")` — it blocks up to 30s waiting for new logs, no need to add delay
3. **ALWAYS show `recent_logs` to the user** — this is the real olive output
4. If status is "running", summarize what olive is doing, then call `get_job_status` again
5. If status is "completed" or "error", show the final result

**Optimization can take 5-30+ minutes depending on model size. This is normal.**

## HuggingFace authentication
Some models (e.g. gated models like meta-llama) require a HuggingFace token to download.
- If a job fails with "401", "403", "authentication", "gated", or "Access denied", **ask the user for their HuggingFace token** and retry with `hf_token`.
- Token from: https://huggingface.co/settings/tokens
- Passed as env var, NOT stored anywhere.

## Tool selection guide (for YOUR decision-making)

### optimize vs quantize for int4
- `optimize` with int4 **always runs GPTQ calibration** — VERY SLOW on CPU (30min+).
- For **fast int4 on CPU**, use `quantize` with `algorithm="rtn"`. Minutes vs hours.
- Only use `optimize` + int4 when user has GPU or explicitly wants GPTQ quality.

### Intent → tool mapping
- **Smallest model / fast inference on CPU** → `quantize(precision="int4", algorithm="rtn")`
- **Smallest model / fast inference on GPU** → `optimize(precision="int4", provider="CUDAExecutionProvider")`
- **Balanced size and quality** → `quantize(precision="int8")`
- **Best quality on GPU** → `optimize(precision="fp16", provider="CUDAExecutionProvider")`
- **Fine-tuning** → `finetune(method="qlora")` (less memory) or `finetune(method="lora")`
- **Just convert to ONNX** → `capture_onnx_graph`

## Popular model recommendations
- **Text chat / general LLM**: microsoft/Phi-4-mini-instruct (small), microsoft/Phi-4 (powerful)
- **Code generation**: microsoft/Phi-4-mini-instruct
- **Image generation**: runwayml/stable-diffusion-v1-5, stabilityai/stable-diffusion-xl-base-1.0
- **Embedding / retrieval**: BAAI/bge-small-en-v1.5, sentence-transformers/all-MiniLM-L6-v2
- **Vision + language**: microsoft/Phi-4-multimodal-instruct
""",
)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

VENV_BASE = Path.home() / ".olive-mcp" / "venvs"
OUTPUT_BASE = Path.home() / ".olive-mcp" / "outputs"
WORKER_PATH = Path(__file__).parent / "worker.py"

# Bump this when _resolve_packages logic changes to invalidate stale cached venvs.
_VENV_CACHE_VERSION = "v2"

# Auto-purge venvs not used within this many days.
_VENV_MAX_AGE_DAYS = 14

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SUPPORTED_DEVICES = ["cpu", "gpu", "npu"]

SUPPORTED_PROVIDERS = [
    "CPUExecutionProvider",
    "CUDAExecutionProvider",
    "DmlExecutionProvider",
    "OpenVINOExecutionProvider",
    "TensorrtExecutionProvider",
    "ROCMExecutionProvider",
    "QNNExecutionProvider",
    "VitisAIExecutionProvider",
    "WebGpuExecutionProvider",
    "NvTensorRTRTXExecutionProvider",
]

SUPPORTED_PRECISIONS = [
    "fp32", "fp16", "bf16",
    "int4", "int8", "int16", "int32",
    "uint4", "uint8", "uint16", "uint32",
]

SUPPORTED_QUANT_ALGORITHMS = ["rtn", "gptq", "awq", "hqq"]

SUPPORTED_QUANT_IMPLEMENTATIONS = ["olive", "ort", "bnb", "nvmo", "inc", "spinquant", "quarot", "awq", "autogptq"]

DEVICE_TO_DEFAULT_PROVIDER = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
    "npu": "QNNExecutionProvider",
}

# Maps provider → olive-ai extras key for onnxruntime variant
PROVIDER_TO_EXTRAS = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "gpu",
    "TensorrtExecutionProvider": "gpu",
    "ROCMExecutionProvider": "gpu",
    "OpenVINOExecutionProvider": "openvino",
    "DmlExecutionProvider": "directml",
    "QNNExecutionProvider": "qnn",
}

# Maps provider → onnxruntime-genai variant (for ModelBuilder pass)
PROVIDER_TO_GENAI = {
    "CPUExecutionProvider": "onnxruntime-genai",
    "CUDAExecutionProvider": "onnxruntime-genai-cuda",
    "DmlExecutionProvider": "onnxruntime-genai-directml",
}


def _resolve_packages(command: str, provider: str | None = None, **kwargs) -> list[str]:
    """Resolve all packages needed for a given olive command + options.

    Uses olive-ai[extras] syntax to pull in the right onnxruntime variant,
    plus any additional packages that specific passes/commands require.
    Includes hidden/implicit dependencies that olive_config.json doesn't declare.
    """
    extras = set()
    extra_packages = []

    # 1. Base ORT variant from provider
    if provider:
        ep_extra = PROVIDER_TO_EXTRAS.get(provider)
        if ep_extra:
            extras.add(ep_extra)

    # 2. Command-specific extras
    if command == "optimize":
        exporter = kwargs.get("exporter") or "model_builder"
        precision = kwargs.get("precision", "fp32")

        if exporter == "model_builder":
            # ModelBuilder pass requires onnxruntime-genai (undeclared in olive_config.json)
            genai_pkg = PROVIDER_TO_GENAI.get(provider or "CPUExecutionProvider", "onnxruntime-genai")
            extra_packages.append(genai_pkg)
        elif exporter == "optimum_exporter":
            extras.add("optimum")

        # GPTQ pass is enabled for int4/uint4 precision → needs calibration data → datasets
        if precision in ("int4", "uint4"):
            extra_packages.append("datasets")

    elif command == "quantize":
        algorithm = kwargs.get("algorithm", "rtn")
        impl = kwargs.get("implementation", "olive")
        if impl == "bnb":
            extras.add("bnb")
        elif impl == "inc":
            extras.add("inc")
        elif impl == "autogptq" or algorithm == "gptq":
            extra_packages.extend(["auto-gptq", "optimum", "datasets"])
        elif impl == "awq" or algorithm == "awq":
            extra_packages.append("autoawq")
        # Static quantization needs calibration data
        if algorithm != "rtn":
            extra_packages.append("datasets")

    elif command == "finetune":
        method = kwargs.get("method", "lora")
        if method == "qlora":
            extras.add("finetune")  # includes bnb, peft, accelerate, etc.
        else:
            extras.add("lora")  # peft, accelerate, scipy
        # Fine-tuning always loads datasets
        extra_packages.append("datasets")

    elif command == "capture_onnx_graph":
        extras.add("capture-onnx-graph")  # optimum only — does NOT include onnxruntime
        # Model builder variant for capture
        if kwargs.get("use_model_builder"):
            genai_pkg = PROVIDER_TO_GENAI.get(provider or "CPUExecutionProvider", "onnxruntime-genai")
            extra_packages.append(genai_pkg)

    elif command == "benchmark":
        device = kwargs.get("device", "cpu")
        if device == "gpu":
            extras.add("gpu")
        else:
            extras.add("cpu")
        # lm-eval is the evaluation backend
        extra_packages.extend(["lm_eval", "datasets"])

    elif command == "diffusion_lora":
        extras.add("diffusers")  # accelerate, peft, diffusers
        extra_packages.append("datasets")

    # 3. Ensure a base onnxruntime is always present.
    #    Many olive passes need onnxruntime at runtime even if they don't declare it.
    #    The extras "cpu", "gpu", "directml", etc. each install the right ORT variant.
    #    If none was added yet, default to "cpu" so bare onnxruntime is available.
    ORT_EXTRAS = {"cpu", "gpu", "directml", "openvino", "qnn"}
    if not extras.intersection(ORT_EXTRAS):
        extras.add("cpu")

    # 4. Build olive-ai install string with extras
    olive_install = f"olive-ai[{','.join(sorted(extras))}]"

    # 5. Deduplicate extra_packages
    seen = set()
    deduped = []
    for pkg in extra_packages:
        if pkg not in seen:
            seen.add(pkg)
            deduped.append(pkg)

    return [olive_install] + deduped


# ---------------------------------------------------------------------------
# Job tracking
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


_JOB_TTL_SECONDS = 3600  # purge finished jobs after 1 hour


def _purge_old_jobs():
    """Remove completed/errored jobs older than _JOB_TTL_SECONDS."""
    now = datetime.now()
    to_delete = [
        jid for jid, job in _jobs.items()
        if job["status"] in ("completed", "error")
        and (now - datetime.fromisoformat(job.get("last_activity", job["started_at"]))).total_seconds() > _JOB_TTL_SECONDS
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


def _get_python_path(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _build_kwargs(**kw) -> dict:
    """Filter out None and False values from kwargs."""
    return {k: v for k, v in kw.items() if v is not None and v is not False}


# ---------------------------------------------------------------------------
# Venv management
# ---------------------------------------------------------------------------


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


async def _get_or_create_venv(packages: list[str], job_id: str) -> Path:
    """Get or create a cached uv venv with the specified packages."""
    # Periodically purge stale venvs
    _purge_old_venvs()

    key = hashlib.md5(f"{_VENV_CACHE_VERSION}|{'|'.join(sorted(packages))}".encode()).hexdigest()[:12]
    venv_path = VENV_BASE / key
    python_path = _get_python_path(venv_path)

    if not python_path.exists():
        _job_log(job_id, f"Creating venv with: {', '.join(packages)}")
        VENV_BASE.mkdir(parents=True, exist_ok=True)

        proc = await asyncio.create_subprocess_exec(
            "uv", "venv", str(venv_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create venv: {stderr.decode()}")

        _job_log(job_id, f"Installing packages: {', '.join(packages)}")
        proc = await asyncio.create_subprocess_exec(
            "uv", "pip", "install", "--python", str(python_path), *packages,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to install packages: {stderr.decode()}")

        _job_log(job_id, "Venv ready")
    else:
        _job_log(job_id, f"Reusing cached venv ({key})")

    _touch_venv(venv_path)
    return python_path


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
        python_path = await _get_or_create_venv(packages, job_id)

        _jobs[job_id]["status"] = "running"
        _job_log(job_id, f"Running: olive {command}")

        # Pass HF token via environment variable (not in kwargs — olive API doesn't take it)
        env = os.environ.copy()
        if hf_token:
            env["HF_TOKEN"] = hf_token
            _job_log(job_id, "HuggingFace token provided")

        proc = await asyncio.create_subprocess_exec(
            str(python_path), "-u", str(WORKER_PATH), command, json.dumps(kwargs),
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
            _job_log(job_id, "Completed successfully" if _jobs[job_id]["status"] == "completed" else "Completed with errors")

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["result"] = {"status": "error", "error": str(exc)}
        _job_log(job_id, f"Exception: {exc}")


_INCOMPATIBLE_PRECISION = {
    "CPUExecutionProvider": {"fp16", "bf16"},
}


def _validate_params(command: str, kwargs: dict) -> str | None:
    """Validate parameters before starting a job. Returns error message or None."""
    provider = kwargs.get("provider")
    precision = kwargs.get("precision")

    # Check provider is recognized
    if provider and provider not in SUPPORTED_PROVIDERS:
        return f"Unknown provider '{provider}'. Supported: {', '.join(SUPPORTED_PROVIDERS)}"

    # Check precision is recognized
    if precision and precision not in SUPPORTED_PRECISIONS:
        return f"Unknown precision '{precision}'. Supported: {', '.join(SUPPORTED_PRECISIONS)}"

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
    if command == "finetune":
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

    job_id = _create_job(command, description)
    packages = _resolve_packages(command, **kwargs)
    asyncio.create_task(
        _run_olive_background(job_id, command, kwargs, packages, hf_token=hf_token)
    )
    return {
        "job_id": job_id,
        "status": "running",
        "packages": packages,
        "message": f"Job started: {description}. Use get_job_status('{job_id}') to check progress.",
    }


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def detect_hardware() -> dict:
    """Detect the user's hardware capabilities (CPU, RAM, GPU, disk space).

    Call this to make smart optimization decisions without asking the user about their device.
    The result tells you what providers and precisions are available on this machine.
    """
    info: dict = {}

    # --- CPU ---
    info["cpu"] = {
        "cores": os.cpu_count(),
        "arch": platform.machine(),
    }

    # --- RAM ---
    try:
        if sys.platform == "win32":
            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]
            mem = MEMORYSTATUSEX()
            mem.dwLength = ctypes.sizeof(MEMORYSTATUSEX)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(mem))
            info["ram"] = {
                "total_gb": round(mem.ullTotalPhys / (1024 ** 3), 1),
                "available_gb": round(mem.ullAvailPhys / (1024 ** 3), 1),
            }
        else:
            # Linux / macOS — read /proc/meminfo or use sysctl
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                lines = meminfo.read_text().splitlines()
                mem_total = mem_avail = 0
                for line in lines:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) * 1024  # kB → bytes
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1]) * 1024
                info["ram"] = {
                    "total_gb": round(mem_total / (1024 ** 3), 1),
                    "available_gb": round(mem_avail / (1024 ** 3), 1),
                }
            else:
                # macOS fallback
                proc = await asyncio.create_subprocess_exec(
                    "sysctl", "-n", "hw.memsize",
                    stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                )
                stdout, _ = await proc.communicate()
                if proc.returncode == 0:
                    total = int(stdout.decode().strip())
                    info["ram"] = {"total_gb": round(total / (1024 ** 3), 1)}
    except Exception:
        info["ram"] = {"error": "Could not detect RAM"}

    # --- GPU (NVIDIA via nvidia-smi) ---
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi", "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            gpus = []
            for line in stdout.decode().strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append({
                        "name": parts[0],
                        "vram_total_mb": int(float(parts[1])),
                        "vram_free_mb": int(float(parts[2])),
                        "driver_version": parts[3],
                    })
            info["gpu"] = {"nvidia": gpus, "cuda_available": len(gpus) > 0}
        else:
            info["gpu"] = {"nvidia": [], "cuda_available": False}
    except FileNotFoundError:
        info["gpu"] = {"nvidia": [], "cuda_available": False}

    # --- Disk space (output directory) ---
    try:
        check_path = OUTPUT_BASE if OUTPUT_BASE.exists() else Path.home()
        usage = shutil.disk_usage(check_path)
        info["disk"] = {
            "total_gb": round(usage.total / (1024 ** 3), 1),
            "free_gb": round(usage.free / (1024 ** 3), 1),
        }
    except Exception:
        info["disk"] = {"error": "Could not detect disk space"}

    # --- OS ---
    info["os"] = {
        "system": platform.system(),
        "version": platform.version(),
    }

    # --- Recommendations ---
    recs = []
    cuda = info.get("gpu", {}).get("cuda_available", False)
    gpus = info.get("gpu", {}).get("nvidia", [])
    ram_gb = info.get("ram", {}).get("total_gb", 0)

    if cuda and gpus:
        vram = gpus[0].get("vram_total_mb", 0)
        recs.append(f"NVIDIA GPU detected ({gpus[0]['name']}, {vram}MB VRAM). Use CUDAExecutionProvider for best performance.")
        if vram >= 8000:
            recs.append("GPU has enough VRAM for fp16 optimization and LoRA fine-tuning.")
        elif vram >= 4000:
            recs.append("GPU VRAM is limited. Prefer int4/int8 quantization. Use QLoRA for fine-tuning.")
    else:
        recs.append("No NVIDIA GPU detected. Use CPUExecutionProvider.")
        recs.append("For CPU: use int4 (smallest) or int8 (balanced). fp16 is NOT supported on CPU.")
        if sys.platform == "win32":
            recs.append("On Windows, you may also try DmlExecutionProvider if you have a DirectX 12 GPU (AMD/Intel/NVIDIA).")

    if ram_gb > 0 and ram_gb < 8:
        recs.append(f"Low RAM ({ram_gb}GB). Prefer small models (e.g. Phi-4-mini) and int4 quantization.")

    info["recommendations"] = recs
    return info


@mcp.tool()
async def get_job_status(job_id: str, last_n_logs: int = 50) -> dict:
    """Check the status and progress of a background job (long-poll).

    This tool **blocks up to 30 seconds** waiting for new olive log output.
    It returns immediately when:
    - New log lines arrive from olive
    - The job completes or errors
    - 30 seconds pass with no new output (returns anyway so you know it's still alive)

    Just call this in a simple loop. Do NOT add any delay — the tool handles timing internally.

    **Show `recent_logs` and `new_lines` to the user every time.** If `new_lines` is 0 and
    `seconds_since_last_output` keeps growing, warn the user the process may be stuck.

    Args:
        job_id: The job ID returned by optimize/quantize/finetune/etc.
        last_n_logs: Number of recent log lines to return. Default: 50.
    """
    if job_id not in _jobs:
        return {"status": "not_found", "error": f"No job with id '{job_id}'"}

    job = _jobs[job_id]

    # Long-poll: wait up to 30 seconds for new logs or job completion.
    new_lines = 0
    if job["status"] in ("starting", "setting_up", "running"):
        prev_count = len(job["log_lines"])
        for _ in range(60):  # 60 x 0.5s = 30 seconds max
            await asyncio.sleep(0.5)
            if job["status"] not in ("starting", "setting_up", "running"):
                break  # job finished
            if len(job["log_lines"]) > prev_count:
                # New logs arrived — wait a tiny bit more to batch them, then return
                await asyncio.sleep(0.5)
                break
        new_lines = len(job["log_lines"]) - prev_count

    now = datetime.now()
    started = datetime.fromisoformat(job["started_at"])
    last_activity = datetime.fromisoformat(job.get("last_activity", job["started_at"]))

    response = {
        "job_id": job_id,
        "status": job["status"],
        "command": job["command"],
        "description": job["description"],
        "elapsed": str(now - started).split(".")[0],  # e.g. "0:05:23"
        "seconds_since_last_output": int((now - last_activity).total_seconds()),
        "new_lines": new_lines,
        "recent_logs": job["log_lines"][-last_n_logs:],
        "total_log_lines": len(job["log_lines"]),
    }

    if job["result"] is not None:
        response["result"] = job["result"]

    return response


@mcp.tool()
async def cancel_job(job_id: str) -> dict:
    """Cancel a running background job.

    Args:
        job_id: The job ID to cancel.
    """
    if job_id not in _jobs:
        return {"status": "not_found", "error": f"No job with id '{job_id}'"}

    job = _jobs[job_id]
    if job["status"] not in ("starting", "setting_up", "running"):
        return {"status": job["status"], "message": f"Job already {job['status']}, cannot cancel."}

    proc = job.get("process")
    if proc and proc.returncode is None:
        proc.terminate()
        _job_log(job_id, "Job cancelled by user")

    job["status"] = "cancelled"
    job["result"] = {"status": "cancelled", "message": "Job was cancelled by user."}
    return {"status": "cancelled", "job_id": job_id, "message": "Job cancelled."}


@mcp.tool()
async def optimize(
    model_name_or_path: str,
    provider: str = "CPUExecutionProvider",
    device: str | None = None,
    precision: str = "fp32",
    act_precision: str | None = None,
    exporter: str | None = None,
    use_qdq_format: bool = False,
    num_split: int | None = None,
    memory: int | None = None,
    block_size: int | None = None,
    surgeries: list[str] | None = None,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Optimize a model end-to-end. Returns a job_id — use get_job_status() to poll progress.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        provider: Execution provider (e.g. "CUDAExecutionProvider"). Default: CPUExecutionProvider.
        device: Target device - "cpu", "gpu", or "npu". Auto-detected from provider if omitted.
        precision: Target precision - "fp32", "fp16", "int4", "int8", etc.
        act_precision: Activation precision for quantization (optional).
        exporter: Model exporter - "model_builder", "dynamo_exporter", "torchscript_exporter", "optimum_exporter".
        use_qdq_format: Use QDQ format for quantization instead of QOperator.
        num_split: Number of splits for model splitting.
        memory: Available device memory in MB.
        block_size: Block size for quantization (-1 for per-channel).
        surgeries: List of graph surgeries to apply.
        output_path: Directory to save optimized model. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not output_path:
        output_path = _make_output_path("optimize", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        provider=provider,
        device=device,
        precision=precision,
        act_precision=act_precision,
        exporter=exporter,
        use_qdq_format=use_qdq_format,
        num_split=num_split,
        memory=memory,
        block_size=block_size,
        surgeries=surgeries,
        output_path=output_path,
    )
    return await _start_job("optimize", f"Optimize {model_name_or_path} ({precision}, {provider})", kwargs, hf_token=hf_token)


@mcp.tool()
async def quantize(
    model_name_or_path: str,
    algorithm: str = "rtn",
    precision: str = "int8",
    act_precision: str = "int8",
    implementation: str = "olive",
    use_qdq_encoding: bool = False,
    data_name: str | None = None,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Quantize a model. Returns a job_id — use get_job_status() to poll progress.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        algorithm: Quantization algorithm - "rtn", "gptq", "awq", "hqq".
        precision: Target precision - "int4", "int8", etc.
        act_precision: Activation precision for static quantization.
        implementation: Backend - "olive", "ort", "bnb", "nvmo", "inc", etc.
        use_qdq_encoding: Use QDQ encoding in ONNX model.
        data_name: HuggingFace dataset name for calibration (required by some algorithms).
        output_path: Directory to save quantized model. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not output_path:
        output_path = _make_output_path("quantize", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        algorithm=algorithm,
        precision=precision,
        act_precision=act_precision,
        implementation=implementation,
        use_qdq_encoding=use_qdq_encoding,
        data_name=data_name,
        output_path=output_path,
    )
    return await _start_job("quantize", f"Quantize {model_name_or_path} ({algorithm}, {precision})", kwargs, hf_token=hf_token)


@mcp.tool()
async def finetune(
    model_name_or_path: str,
    data_name: str,
    method: str = "lora",
    lora_r: int = 64,
    lora_alpha: int = 16,
    target_modules: str | None = None,
    torch_dtype: str = "bfloat16",
    train_split: str = "train",
    eval_split: str | None = None,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Fine-tune a model using LoRA or QLoRA. Returns a job_id — use get_job_status() to poll.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "microsoft/Phi-3-mini-4k-instruct").
        data_name: HuggingFace dataset name (e.g. "nampdn-ai/tiny-codes").
        method: Fine-tuning method - "lora" or "qlora" (4-bit quantized, less memory).
        lora_r: LoRA rank. Default: 64.
        lora_alpha: LoRA alpha scaling. Default: 16.
        target_modules: Comma-separated target modules for LoRA.
        torch_dtype: Torch dtype for training - "bfloat16", "float16", "float32".
        train_split: Dataset split for training. Default: "train".
        eval_split: Dataset split for evaluation (optional).
        output_path: Directory to save fine-tuned adapter. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not output_path:
        output_path = _make_output_path("finetune", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        data_name=data_name,
        method=method,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=target_modules,
        torch_dtype=torch_dtype,
        train_split=train_split,
        eval_split=eval_split,
        output_path=output_path,
    )
    return await _start_job("finetune", f"Finetune {model_name_or_path} ({method}) on {data_name}", kwargs, hf_token=hf_token)


@mcp.tool()
async def capture_onnx_graph(
    model_name_or_path: str,
    use_model_builder: bool = False,
    use_dynamo_exporter: bool = False,
    precision: str = "fp16",
    conversion_device: str = "cpu",
    torch_dtype: str | None = None,
    target_opset: int = 20,
    use_ort_genai: bool = False,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Capture ONNX graph from a model. Returns a job_id — use get_job_status() to poll.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        use_model_builder: Use Model Builder to capture ONNX model.
        use_dynamo_exporter: Use dynamo_export API to export ONNX model.
        precision: Precision for Model Builder - "fp16", "fp32", "int4", "bf16".
        conversion_device: Device for conversion - "cpu" or "gpu".
        torch_dtype: Dtype to cast model before capture (e.g. "float16").
        target_opset: Target ONNX opset version. Default: 20.
        use_ort_genai: Use ORT generate() API to run the model.
        output_path: Directory to save ONNX model. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not output_path:
        output_path = _make_output_path("capture", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        use_model_builder=use_model_builder,
        use_dynamo_exporter=use_dynamo_exporter,
        precision=precision,
        conversion_device=conversion_device,
        torch_dtype=torch_dtype,
        target_opset=target_opset,
        use_ort_genai=use_ort_genai,
        output_path=output_path,
    )
    return await _start_job("capture_onnx_graph", f"Capture ONNX from {model_name_or_path}", kwargs, hf_token=hf_token)


@mcp.tool()
async def benchmark(
    model_name_or_path: str,
    tasks: list[str] | None = None,
    device: str = "cpu",
    batch_size: int = 1,
    max_length: int = 1024,
    limit: float = 1.0,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Benchmark/evaluate a model. Returns a job_id — use get_job_status() to poll.

    Args:
        model_name_or_path: HuggingFace model name or local path.
        tasks: List of lm-eval tasks. Default: ["hellaswag"].
        device: Device for evaluation - "cpu" or "gpu".
        batch_size: Evaluation batch size. Default: 1.
        max_length: Maximum length of input + output. Default: 1024.
        limit: Fraction of samples (0.0-1.0) or absolute number. Default: 1.0.
        output_path: Directory to save results. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not tasks:
        tasks = ["hellaswag"]
    if not output_path:
        output_path = _make_output_path("benchmark", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        limit=limit,
        output_path=output_path,
    )
    return await _start_job("benchmark", f"Benchmark {model_name_or_path} on {', '.join(tasks)}", kwargs, hf_token=hf_token)


@mcp.tool()
async def diffusion_lora(
    model_name_or_path: str,
    data_dir: str | None = None,
    data_name: str | None = None,
    model_variant: str = "auto",
    lora_r: int = 16,
    alpha: float | None = None,
    max_train_steps: int = 1000,
    learning_rate: float = 1e-4,
    train_batch_size: int = 1,
    mixed_precision: str = "bf16",
    dreambooth: bool = False,
    instance_prompt: str | None = None,
    merge_lora: bool = False,
    output_path: str | None = None,
    hf_token: str | None = None,
) -> dict:
    """Train LoRA for diffusion models. Returns a job_id — use get_job_status() to poll.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "runwayml/stable-diffusion-v1-5").
        data_dir: Path to local image folder with training images.
        data_name: HuggingFace dataset name (alternative to data_dir).
        model_variant: Model type - "auto", "sd", "sdxl", "flux".
        lora_r: LoRA rank. SD: 4-16, Flux: 16-64. Default: 16.
        alpha: LoRA alpha for scaling. Default: same as r.
        max_train_steps: Maximum training steps. Default: 1000.
        learning_rate: Learning rate. Default: 1e-4.
        train_batch_size: Training batch size. Default: 1.
        mixed_precision: Mixed precision - "bf16", "fp16", "no". Default: bf16.
        dreambooth: Enable DreamBooth training.
        instance_prompt: Fixed prompt for DreamBooth mode.
        merge_lora: Merge LoRA into base model instead of saving adapter only.
        output_path: Directory to save LoRA adapter. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.
    """
    if not output_path:
        output_path = _make_output_path("diffusion_lora", model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        data_dir=data_dir,
        data_name=data_name,
        model_variant=model_variant,
        lora_r=lora_r,
        alpha=alpha,
        max_train_steps=max_train_steps,
        learning_rate=learning_rate,
        train_batch_size=train_batch_size,
        mixed_precision=mixed_precision,
        dreambooth=dreambooth,
        instance_prompt=instance_prompt,
        merge_lora=merge_lora,
        output_path=output_path,
    )
    return await _start_job("diffusion_lora", f"Diffusion LoRA for {model_name_or_path}", kwargs, hf_token=hf_token)


@mcp.tool()
async def manage_outputs(
    action: str = "list",
    prefix: str | None = None,
    names: list[str] | None = None,
    delete_all: bool = False,
    limit: int = 20,
) -> dict:
    """List or delete previous optimization outputs saved by olive-mcp.

    Actions:
    - `list` (default): Show past optimization runs. Use `prefix` to filter by operation type.
    - `delete`: Delete specific outputs. Specify `names`, `prefix`, or `delete_all`.
      When deleting, always list first to confirm with the user.

    Args:
        action: "list" to browse outputs, "delete" to remove them.
        prefix: Filter by operation type (e.g. "optimize", "quantize", "finetune").
        names: List of output directory names to delete (from a previous list action).
        delete_all: Set to true to delete ALL outputs (use with action="delete").
        limit: Maximum number of results to return when listing. Default: 20.
    """
    if not OUTPUT_BASE.exists():
        return {"outputs": [], "message": "No outputs found. Run an optimization first."}

    # --- DELETE ---
    if action == "delete":
        if not names and not prefix and not delete_all:
            return {"deleted": 0, "error": "Specify names, prefix, or delete_all=true."}

        deleted = []
        errors = []
        for d in list(OUTPUT_BASE.iterdir()):
            if not d.is_dir():
                continue
            should_delete = False
            if delete_all:
                should_delete = True
            elif names and d.name in names:
                should_delete = True
            elif prefix and d.name.startswith(prefix + "_"):
                should_delete = True
            if should_delete:
                try:
                    shutil.rmtree(d)
                    deleted.append(d.name)
                except Exception as e:
                    errors.append({"name": d.name, "error": str(e)})

        result = {"deleted": len(deleted), "deleted_names": deleted}
        if errors:
            result["errors"] = errors
        return result

    # --- LIST ---
    _KNOWN_PREFIXES = [
        "diffusion_lora", "capture_onnx_graph",
        "optimize", "quantize", "finetune", "capture", "benchmark",
    ]

    entries = []
    for d in sorted(OUTPUT_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue

        entry = {"name": d.name, "path": str(d)}

        for op in _KNOWN_PREFIXES:
            if d.name.startswith(op + "_"):
                entry["operation"] = op
                tail = d.name.rsplit("_", 2)
                if len(tail) >= 3:
                    entry["timestamp"] = f"{tail[-2]}_{tail[-1]}"
                break

        # Find model files
        for ext in ("*.onnx", "*.pt", "*.safetensors", "*.bin"):
            files = list(d.rglob(ext))
            if files:
                entry["model_files"] = [str(f) for f in files[:5]]
                break

        # Check for config
        for cfg_name in ("model_config.json", "config.json", "inference_config.json"):
            cfg = d / cfg_name
            if cfg.exists():
                entry["has_config"] = True
                break

        entries.append(entry)
        if len(entries) >= limit:
            break

    return {"outputs": entries, "total": len(entries)}


# ---------------------------------------------------------------------------
# MCP Prompts — guided workflows for beginners
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="optimize-model",
    description="Guided model optimization — detects your hardware and picks the best settings automatically.",
)
def prompt_optimize_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to optimize a model using Olive. Help me:\n"
                "1. Run `detect_hardware` to see what I have (GPU, RAM, etc.)\n"
                "2. Ask what I want to use the model for (chat, code, images, etc.)\n"
                "3. Based on my hardware and use case, recommend a model and settings\n"
                "4. Give me a few plain-language options (smallest, balanced, best quality)\n"
                "5. Run the optimization based on my choice\n"
                "6. Show me the results and suggest next steps"
            ),
        }
    ]


@mcp.prompt(
    name="quantize-model",
    description="Guided model quantization — makes your model smaller and faster.",
)
def prompt_quantize_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to quantize a model to make it smaller/faster. Help me:\n"
                "1. Run `detect_hardware` to see what I have\n"
                "2. Ask which model I want to quantize (or recommend one for my use case)\n"
                "3. Based on my hardware, pick the best precision and algorithm\n"
                "4. Run the quantization\n"
                "5. Show me the results (size reduction, etc.)"
            ),
        }
    ]


@mcp.prompt(
    name="finetune-model",
    description="Guided model fine-tuning — sets up LoRA/QLoRA training based on your hardware.",
)
def prompt_finetune_model() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to fine-tune a model on my own data. Help me:\n"
                "1. Run `detect_hardware` to check my GPU and memory\n"
                "2. Ask what task I want the model to learn\n"
                "3. Ask about my training data (HuggingFace dataset name or local path)\n"
                "4. Based on my hardware, pick LoRA or QLoRA and recommend a base model\n"
                "5. Run fine-tuning\n"
                "6. Suggest next steps (optimize the fine-tuned model, benchmark, etc.)"
            ),
        }
    ]


@mcp.prompt(
    name="compare-models",
    description="Compare previous optimization results — find the best model from past runs.",
)
def prompt_compare_models() -> list[dict]:
    return [
        {
            "role": "user",
            "content": (
                "I want to compare my previous optimization results. Help me:\n"
                "1. Use `manage_outputs` to show my past optimization runs\n"
                "2. Summarize each run (model, precision, size, metrics if available)\n"
                "3. Recommend which result is best for my needs\n"
                "4. If I haven't benchmarked yet, suggest running `benchmark` on the candidates"
            ),
        }
    ]
