"""Olive Config MCP Server - Pass exploration and config generation/validation."""

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
from typing import Any

from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    name="olive-config",
    instructions="""Olive Config MCP server for advanced config file generation and validation.

This server helps users write valid Olive workflow config files by exploring available
passes, their parameters, and validating configs before running them.

## When to use this server
Use this when the user wants to:
- Write a custom Olive config file with specific passes
- Understand what passes are available and what they do
- Chain multiple passes together in a specific order
- Fine-tune pass parameters beyond what the high-level CLI offers

## Workflow

### If the user gives a specific request (e.g. "create a config with OnnxQuantization + GraphSurgeries"):
- Call `explore_passes(pass_name=X)` for each pass to get their parameter schemas
- Generate the config JSON directly
- Validate with `run_config(config, validate_only=true)`
- If errors, fix and re-validate
- **Show the final config to the user and ask for confirmation before running**

### If the user gives a vague request (e.g. "help me write an olive config"):
- Ask ONE question: what model + what they want to achieve
- Call `detect_hardware` to auto-detect GPU/CPU — do NOT ask the user about their device
- Call `explore_passes()` filtered by their target to show relevant passes
- Recommend a pass pipeline based on their goal
- Generate config with the correct execution provider based on hardware detection
- Validate, and show to user
- **Wait for user confirmation before running**

## Hardware detection
Before generating any config, call `detect_hardware` to determine:
- Whether the user has a GPU → use CUDAExecutionProvider / DmlExecutionProvider
- CPU only → use CPUExecutionProvider, avoid fp16 passes
- Available VRAM → recommend appropriate model sizes and precisions
Never ask the user "do you have a GPU?" — just detect it.

## IMPORTANT: Always show config before running
Before calling `run_config(config, validate_only=false)`, you MUST:
1. Show the complete config JSON to the user
2. Explain what each pass does and why it's included
3. Wait for the user to confirm they want to run it
Never auto-run a config without user approval — runs can take minutes to hours and consume significant resources.

## Config file structure
```json
{
  "input_model": {
    "type": "HfModel",
    "model_path": "microsoft/Phi-4-mini-instruct"
  },
  "systems": {
    "local": {
      "type": "LocalSystem",
      "config": { "accelerators": [{"device": "cpu"}] }
    }
  },
  "passes": {
    "pass_name": [{"type": "PassClassName", "config": {...}}]
  },
  "engine": {
    "host": "local",
    "target": "local"
  }
}
```

Passes run in the order they appear in the config. Each pass's output becomes the next pass's input.

## Common pass pipelines
- **Quantize for CPU**: OnnxConversion → OnnxQuantization (or OnnxBlockWiseRtnQuantization for int4)
- **Optimize for GPU**: OnnxConversion → OrtTransformersOptimization → OnnxFloatToFloat16
- **Fine-tune + export**: LoRA → OnnxConversion → OnnxQuantization
- **Model Builder (LLM)**: ModelBuilder (handles conversion + optimization in one pass)
""",
)

# ---------------------------------------------------------------------------
# Paths & venv config
# ---------------------------------------------------------------------------

VENV_BASE = Path.home() / ".olive-mcp" / "venvs"
_VENV_CACHE_VERSION = "v2"
_VENV_MAX_AGE_DAYS = 14

# Maps execution provider → olive-ai extras key for onnxruntime variant
PROVIDER_TO_EXTRAS = {
    "CPUExecutionProvider": "cpu",
    "CUDAExecutionProvider": "gpu",
    "TensorrtExecutionProvider": "gpu",
    "ROCMExecutionProvider": "gpu",
    "OpenVINOExecutionProvider": "openvino",
    "DmlExecutionProvider": "directml",
    "QNNExecutionProvider": "qnn",
}


# ---------------------------------------------------------------------------
# Olive imports (loaded at startup so tool calls are instant)
# ---------------------------------------------------------------------------

from olive.package_config import OlivePackageConfig  # noqa: E402
from olive.passes.olive_pass import Pass  # noqa: E402

_olive_config = OlivePackageConfig.load_default_config()
_pass_registry = Pass.registry


def _get_pass_module_config(pass_name: str):
    """Get PassModuleConfig from olive_config.json by name (case-insensitive)."""

    # olive_config.json keys are PascalCase, try exact match first
    for key in _olive_config.passes:
        if key.lower() == pass_name.lower():
            return key, _olive_config.passes[key]
    return None, None


def _get_pass_class(pass_name: str):
    """Get pass class by name. Tries registry first, then dynamic import via module_path."""

    # Try registry first (already imported passes)
    cls = _pass_registry.get(pass_name.lower())
    if cls:
        return cls

    # Dynamic import using module_path from olive_config.json
    _, module_config = _get_pass_module_config(pass_name)
    if module_config and module_config.module_path:
        import importlib

        # module_path is like "olive.passes.onnx.quantization.OnnxQuantization"
        parts = module_config.module_path.rsplit(".", 1)
        if len(parts) == 2:
            try:
                mod = importlib.import_module(parts[0])
                cls = getattr(mod, parts[1], None)
                return cls
            except Exception:
                pass
    return None


def _serialize_type(type_annotation) -> str:
    """Convert a type annotation to a readable string."""
    if type_annotation is None:
        return "any"
    name = getattr(type_annotation, "__name__", None)
    if name:
        return name
    return str(type_annotation).replace("typing.", "")


def _serialize_default(value) -> Any:
    """Convert a default value to something JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_default(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_default(v) for k, v in value.items()}
    # For complex objects (ConditionalDefault, SearchParameter, etc.)
    return str(value)


# ---------------------------------------------------------------------------
# Venv management
# ---------------------------------------------------------------------------


def _get_python_path(venv_path: Path) -> Path:
    if sys.platform == "win32":
        return venv_path / "Scripts" / "python.exe"
    return venv_path / "bin" / "python"


def _resolve_packages_from_config(config: dict) -> list[str]:
    """Parse an Olive config dict to determine required packages.

    Looks at the systems section to find execution providers, then maps
    them to the correct olive-ai extras (cpu, gpu, directml, etc.).
    """
    extras = set()

    # Extract execution providers from systems → accelerators
    systems = config.get("systems", {})
    for sys_config in systems.values():
        inner = sys_config.get("config", {})
        for acc in inner.get("accelerators", []):
            ep = acc.get("execution_provider")
            if ep and ep in PROVIDER_TO_EXTRAS:
                extras.add(PROVIDER_TO_EXTRAS[ep])
            # Also check device as fallback
            device = acc.get("device", "").lower()
            if not ep:
                if device == "gpu":
                    extras.add("gpu")
                elif device == "npu":
                    extras.add("qnn")

    # Ensure at least a base onnxruntime is present
    ORT_EXTRAS = {"cpu", "gpu", "directml", "openvino", "qnn"}
    if not extras.intersection(ORT_EXTRAS):
        extras.add("cpu")

    return [f"olive-ai[{','.join(sorted(extras))}]"]


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
            age = (now - datetime.fromtimestamp(d.stat().st_mtime)).days
        if age > _VENV_MAX_AGE_DAYS:
            shutil.rmtree(d, ignore_errors=True)


def _touch_venv(venv_path: Path):
    """Update the last-used marker for a cached venv."""
    marker = venv_path / ".last_used"
    marker.touch()


async def _get_or_create_venv(packages: list[str], job_id: str) -> Path:
    """Get or create a cached uv venv with the specified packages."""
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
# Job tracking (for run_config async execution)
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}


def _create_job(description: str) -> str:
    job_id = uuid.uuid4().hex[:8]
    _jobs[job_id] = {
        "status": "running",
        "description": description,
        "log_lines": [],
        "result": None,
        "process": None,
        "started_at": datetime.now().isoformat(),
    }
    return job_id


def _job_log(job_id: str, line: str):
    if job_id in _jobs:
        _jobs[job_id]["log_lines"].append(line)
        _jobs[job_id]["last_activity"] = datetime.now().isoformat()


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool()
async def detect_hardware() -> dict:
    """Detect the user's hardware capabilities (CPU, RAM, GPU, disk space).

    Call this before generating a config to determine the right execution provider
    and passes. Returns hardware info plus recommendations for provider and precision.
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
            meminfo = Path("/proc/meminfo")
            if meminfo.exists():
                lines = meminfo.read_text().splitlines()
                mem_total = mem_avail = 0
                for line in lines:
                    if line.startswith("MemTotal:"):
                        mem_total = int(line.split()[1]) * 1024
                    elif line.startswith("MemAvailable:"):
                        mem_avail = int(line.split()[1]) * 1024
                info["ram"] = {
                    "total_gb": round(mem_total / (1024 ** 3), 1),
                    "available_gb": round(mem_avail / (1024 ** 3), 1),
                }
            else:
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

    # --- Disk space ---
    try:
        usage = shutil.disk_usage(Path.home())
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

    # --- Recommendations for config generation ---
    recs = []
    cuda = info.get("gpu", {}).get("cuda_available", False)
    gpus = info.get("gpu", {}).get("nvidia", [])
    ram_gb = info.get("ram", {}).get("total_gb", 0)

    if cuda and gpus:
        vram = gpus[0].get("vram_total_mb", 0)
        recs.append(f"NVIDIA GPU detected ({gpus[0]['name']}, {vram}MB VRAM). Use CUDAExecutionProvider.")
        if vram >= 8000:
            recs.append("Enough VRAM for fp16 optimization and large models.")
        elif vram >= 4000:
            recs.append("Limited VRAM. Prefer int4/int8 quantization passes.")
        info["recommended_provider"] = "CUDAExecutionProvider"
        info["recommended_device"] = "gpu"
    else:
        recs.append("No NVIDIA GPU detected. Use CPUExecutionProvider.")
        recs.append("CPU does NOT support fp16. Use int4 or int8 quantization passes.")
        if sys.platform == "win32":
            recs.append("On Windows, DmlExecutionProvider may work if you have a DirectX 12 GPU (AMD/Intel/NVIDIA).")
        info["recommended_provider"] = "CPUExecutionProvider"
        info["recommended_device"] = "cpu"

    if ram_gb > 0 and ram_gb < 8:
        recs.append(f"Low RAM ({ram_gb}GB). Prefer small models and int4 quantization.")

    info["recommendations"] = recs
    return info


@mcp.tool()
async def explore_passes(
    pass_name: str | None = None,
    provider: str | None = None,
    precision: str | None = None,
    accelerator: str | None = None,
) -> dict:
    """Explore available Olive passes and their parameter schemas.

    Two modes:
    - **List mode** (no pass_name): List all passes, optionally filtered by provider/precision/accelerator.
      Returns pass names with their supported configurations and dataset requirements.
    - **Detail mode** (with pass_name): Get full parameter schema for a specific pass,
      including each parameter's type, default value, whether it's required, and description.

    Args:
        pass_name: If provided, get full parameter schema for this pass. If omitted, list all passes.
        provider: Filter passes by execution provider (e.g. "CPUExecutionProvider", "CUDAExecutionProvider").
        precision: Filter passes by supported precision (e.g. "int4", "int8", "fp16").
        accelerator: Filter passes by accelerator type (e.g. "cpu", "gpu", "npu").
    """


    # --- Detail mode: get full schema for one pass ---
    if pass_name:
        canonical_name, module_config = _get_pass_module_config(pass_name)
        if not module_config:
            available = sorted(_olive_config.passes.keys())
            return {"error": f"Pass '{pass_name}' not found. Available: {available}"}

        result = {
            "name": canonical_name,
            "module_path": module_config.module_path,
            "supported_providers": list(module_config.supported_providers),
            "supported_accelerators": list(module_config.supported_accelerators),
            "supported_precisions": list(module_config.supported_precisions),
            "supported_algorithms": list(module_config.supported_algorithms or []),
            "dataset": module_config.dataset,
            "parameters": {},
        }

        # Try to get the full parameter schema from the pass class
        pass_cls = _get_pass_class(pass_name)
        if pass_cls:
            try:
                from olive.hardware import AcceleratorSpec, Device
                from olive.hardware.constants import ExecutionProvider

                spec = AcceleratorSpec(
                    accelerator_type=Device.CPU,
                    execution_provider=ExecutionProvider.CPUExecutionProvider,
                )
                default_config = pass_cls._default_config(spec)

                for param_name, param in default_config.items():
                    try:
                        result["parameters"][param_name] = {
                            "type": _serialize_type(getattr(param, "type_", None)),
                            "required": getattr(param, "required", False),
                            "default": _serialize_default(getattr(param, "default_value", None)),
                            "description": getattr(param, "description", "") or "",
                        }
                    except Exception:
                        result["parameters"][param_name] = {
                            "type": "unknown",
                            "required": False,
                            "default": None,
                            "description": "",
                        }
            except Exception as e:
                result["schema_error"] = f"Could not load parameter schema: {e}"

        return result

    # --- List mode: list all passes with filtering ---
    passes = []
    for name, module_config in sorted(_olive_config.passes.items()):
        # Apply filters (wildcard "*" means supports everything)
        if provider:
            supported = module_config.supported_providers
            if "*" not in supported and provider not in supported:
                continue
        if precision:
            supported = module_config.supported_precisions
            if "*" not in supported and precision not in supported:
                continue
        if accelerator:
            supported = module_config.supported_accelerators
            if "*" not in supported and accelerator not in supported:
                continue

        passes.append({
            "name": name,
            "providers": list(module_config.supported_providers),
            "accelerators": list(module_config.supported_accelerators),
            "precisions": list(module_config.supported_precisions),
            "algorithms": list(module_config.supported_algorithms or []),
            "dataset": module_config.dataset,
        })

    return {"passes": passes, "total": len(passes)}


@mcp.tool()
async def run_config(
    config: dict,
    validate_only: bool = False,
    output_path: str | None = None,
) -> dict:
    """Validate or run an Olive workflow config.

    Use `validate_only=true` to check if a config is valid without running it.
    Always validate first before running, to catch errors instantly instead of
    waiting for a long-running job to fail.

    Args:
        config: The Olive workflow config as a JSON object.
        validate_only: If true, only validate the config. If false, validate and run.
        output_path: Directory to save results (auto-generated if omitted). Ignored when validate_only=true.
    """

    # --- Validation ---
    try:
        from olive.workflows.run.config import RunConfig

        RunConfig.parse_file_or_obj(config)
    except Exception as e:
        error_str = str(e)
        # Try to extract structured errors from Pydantic ValidationError
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                errors.append({"location": loc, "message": err.get("msg", ""), "type": err.get("type", "")})

        return {
            "valid": False,
            "errors": errors if errors else [{"message": error_str}],
        }

    if validate_only:
        return {"valid": True, "message": "Config is valid."}

    # --- Run ---
    if not output_path:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = str(Path.home() / ".olive-mcp" / "config-outputs" / f"run_{ts}")
        Path(output_path).mkdir(parents=True, exist_ok=True)

    # Write config to temp file
    config_file = Path(output_path) / "olive_config.json"
    config_file.write_text(json.dumps(config, indent=2))

    # Resolve packages from config
    packages = _resolve_packages_from_config(config)

    job_id = _create_job(f"Run config → {output_path}")
    _job_log(job_id, f"Config written to {config_file}")
    _job_log(job_id, f"Packages: {', '.join(packages)}")

    # Launch olive in background
    asyncio.create_task(_run_config_background(job_id, str(config_file), output_path, packages))

    return {
        "job_id": job_id,
        "status": "running",
        "config_file": str(config_file),
        "output_path": output_path,
        "message": f"Job started. Use get_job_status('{job_id}') to check progress.",
    }


async def _run_config_background(job_id: str, config_file: str, output_path: str, packages: list[str]):
    """Run olive config in isolated venv."""
    try:
        # Create or reuse venv with correct packages
        python_path = await _get_or_create_venv(packages, job_id)

        _job_log(job_id, "Starting olive run...")
        proc = await asyncio.create_subprocess_exec(
            str(python_path), "-m", "olive", "run",
            "--config", config_file, "--output-path", output_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _jobs[job_id]["process"] = proc

        # Stream stderr (olive logs)
        while True:
            try:
                line = await proc.stderr.readline()
            except ValueError:
                continue
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            if decoded:
                if len(decoded) > 500:
                    decoded = decoded[:500] + "... (truncated)"
                _job_log(job_id, decoded)

        stdout_bytes = await proc.stdout.read()
        await proc.wait()

        if _jobs[job_id]["status"] == "cancelled":
            return

        if proc.returncode != 0:
            _jobs[job_id]["status"] = "error"
            _jobs[job_id]["result"] = {
                "status": "error",
                "returncode": proc.returncode,
                "output": stdout_bytes.decode("utf-8", errors="replace")[-3000:],
            }
            _job_log(job_id, f"Failed with exit code {proc.returncode}")
        else:
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = {
                "status": "success",
                "output_path": output_path,
            }
            _job_log(job_id, "Completed successfully")

    except Exception as exc:
        _jobs[job_id]["status"] = "error"
        _jobs[job_id]["result"] = {"status": "error", "error": str(exc)}
        _job_log(job_id, f"Exception: {exc}")


@mcp.tool()
async def get_job_status(job_id: str, last_n_logs: int = 50) -> dict:
    """Check the status of a running config job (long-poll).

    Blocks up to 30 seconds waiting for new log output. Returns immediately when
    new logs arrive or the job completes.

    Args:
        job_id: The job ID returned by run_config.
        last_n_logs: Number of recent log lines to return. Default: 50.
    """
    if job_id not in _jobs:
        return {"status": "not_found", "error": f"No job with id '{job_id}'"}

    job = _jobs[job_id]

    # Long-poll
    new_lines = 0
    if job["status"] == "running":
        prev_count = len(job["log_lines"])
        for _ in range(60):
            await asyncio.sleep(0.5)
            if job["status"] != "running":
                break
            if len(job["log_lines"]) > prev_count:
                await asyncio.sleep(0.5)
                break
        new_lines = len(job["log_lines"]) - prev_count

    now = datetime.now()
    started = datetime.fromisoformat(job["started_at"])
    last_activity = datetime.fromisoformat(job.get("last_activity", job["started_at"]))

    response = {
        "job_id": job_id,
        "status": job["status"],
        "description": job["description"],
        "elapsed": str(now - started).split(".")[0],
        "seconds_since_last_output": int((now - last_activity).total_seconds()),
        "new_lines": new_lines,
        "recent_logs": job["log_lines"][-last_n_logs:],
        "total_log_lines": len(job["log_lines"]),
    }

    if job["result"] is not None:
        response["result"] = job["result"]

    return response
