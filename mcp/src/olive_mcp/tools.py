# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import asyncio
import os
import platform
import shutil
import sys
from datetime import datetime
from pathlib import Path

import psutil

from olive_mcp.constants import (
    OUTPUT_BASE,
    Command,
)
from olive_mcp.jobs import (
    _build_kwargs,
    _detect_phase,
    _job_log,
    _jobs,
    _make_output_path,
    _start_job,
)
from olive_mcp.server import mcp

_DIFFUSION_KEYWORDS = ("stable-diffusion", "sdxl", "flux", "diffusion")


def _is_diffusion_model(model_name_or_path: str) -> bool:
    """Detect if a model is a diffusion model based on its name."""
    return any(k in model_name_or_path.lower() for k in _DIFFUSION_KEYWORDS)


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
        mem = psutil.virtual_memory()
        info["ram"] = {
            "total_gb": round(mem.total / (1024**3), 1),
            "available_gb": round(mem.available / (1024**3), 1),
        }
    except Exception:
        info["ram"] = {"error": "Could not detect RAM"}

    # --- GPU (NVIDIA via nvidia-smi) ---
    try:
        proc = await asyncio.create_subprocess_exec(
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version",
            "--format=csv,noheader,nounits",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await proc.communicate()
        if proc.returncode == 0:
            gpus = []
            for line in stdout.decode().strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 4:
                    gpus.append(
                        {
                            "name": parts[0],
                            "vram_total_mb": int(float(parts[1])),
                            "vram_free_mb": int(float(parts[2])),
                            "driver_version": parts[3],
                        }
                    )
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
            "total_gb": round(usage.total / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
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
        recs.append(
            f"NVIDIA GPU detected ({gpus[0]['name']}, {vram}MB VRAM). Use CUDAExecutionProvider for best performance."
        )
        if vram >= 8000:
            recs.append("GPU has enough VRAM for fp16 optimization and LoRA fine-tuning.")
        elif vram >= 4000:
            recs.append("GPU VRAM is limited. Prefer int4/int8 quantization. Use QLoRA for fine-tuning.")
    else:
        recs.append("No NVIDIA GPU detected. Use CPUExecutionProvider.")
        recs.append("For CPU: use int4 (smallest) or int8 (balanced). fp16 is NOT supported on CPU.")
        if sys.platform == "win32":
            recs.append(
                "On Windows, you may also try DmlExecutionProvider if you have a DirectX 12 GPU (AMD/Intel/NVIDIA)."
            )

    if 0 < ram_gb < 8:
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
        "elapsed": str(now - started).split(".", maxsplit=1)[0],  # e.g. "0:05:23"
        "seconds_since_last_output": int((now - last_activity).total_seconds()),
        "new_lines": new_lines,
        "recent_logs": job["log_lines"][-last_n_logs:],
        "total_log_lines": len(job["log_lines"]),
    }

    # Structured phase detection for active jobs
    if job["status"] in ("starting", "setting_up", "running") and job["log_lines"]:
        phase_info = _detect_phase(job["log_lines"])
        response["phase"] = phase_info["phase"]
        response["phase_description"] = phase_info["phase_description"]
        if "current_pass" in phase_info:
            response["current_pass"] = phase_info["current_pass"]

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
        if sys.platform == "win32":
            import subprocess as _sp

            _sp.call(["taskkill", "/F", "/T", "/PID", str(proc.pid)], stdout=_sp.DEVNULL, stderr=_sp.DEVNULL)
        else:
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
        output_path = _make_output_path(Command.OPTIMIZE, model_name_or_path)

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
    return await _start_job(
        Command.OPTIMIZE, f"Optimize {model_name_or_path} ({precision}, {provider})", kwargs, hf_token=hf_token
    )


@mcp.tool()
async def quantize(
    model_name_or_path: str,
    algorithm: str = "rtn",
    precision: str = "int8",
    act_precision: str | None = None,
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
        output_path = _make_output_path(Command.QUANTIZE, model_name_or_path)

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
    return await _start_job(
        Command.QUANTIZE, f"Quantize {model_name_or_path} ({algorithm}, {precision})", kwargs, hf_token=hf_token
    )


@mcp.tool()
async def finetune(
    model_name_or_path: str,
    data_name: str | None = None,
    method: str = "lora",
    lora_r: int = 64,
    lora_alpha: int = 16,
    target_modules: str | None = None,
    torch_dtype: str = "bfloat16",
    train_split: str = "train",
    eval_split: str | None = None,
    data_dir: str | None = None,
    model_variant: str = "auto",
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
    """Fine-tune a model using LoRA/QLoRA. Returns a job_id — use get_job_status() to poll.

    Automatically detects diffusion models (Stable Diffusion, SDXL, Flux) and routes
    to the appropriate training backend.

    Args:
        model_name_or_path: HuggingFace model name (e.g. "microsoft/Phi-3-mini-4k-instruct"
            for text, "runwayml/stable-diffusion-v1-5" for diffusion).
        data_name: HuggingFace dataset name. Required for text models, optional for diffusion.
        method: [Text only] Fine-tuning method - "lora" or "qlora" (4-bit quantized, less memory).
        lora_r: LoRA rank. Default: 64.
        lora_alpha: [Text only] LoRA alpha scaling. Default: 16.
        target_modules: [Text only] Comma-separated target modules for LoRA.
        torch_dtype: [Text only] Torch dtype for training - "bfloat16", "float16", "float32".
        train_split: [Text only] Dataset split for training. Default: "train".
        eval_split: [Text only] Dataset split for evaluation (optional).
        data_dir: [Diffusion only] Path to local image folder with training images.
        model_variant: [Diffusion only] Model type - "auto", "sd", "sdxl", "flux".
        alpha: [Diffusion only] LoRA alpha for scaling. Default: same as lora_r.
        max_train_steps: [Diffusion only] Maximum training steps. Default: 1000.
        learning_rate: [Diffusion only] Learning rate. Default: 1e-4.
        train_batch_size: [Diffusion only] Training batch size. Default: 1.
        mixed_precision: [Diffusion only] Mixed precision - "bf16", "fp16", "no". Default: bf16.
        dreambooth: [Diffusion only] Enable DreamBooth training.
        instance_prompt: [Diffusion only] Fixed prompt for DreamBooth mode.
        merge_lora: [Diffusion only] Merge LoRA into base model instead of saving adapter only.
        output_path: Directory to save fine-tuned adapter. Auto-generated if omitted.
        hf_token: HuggingFace token for gated/private models. Ask the user if download fails with 401/403.

    """
    if _is_diffusion_model(model_name_or_path):
        if not data_dir and not data_name:
            return {
                "status": "error",
                "error": "Either data_dir or data_name is required for diffusion model fine-tuning.",
            }
        if not output_path:
            output_path = _make_output_path(Command.DIFFUSION_LORA, model_name_or_path)
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
        return await _start_job(
            Command.DIFFUSION_LORA, f"Diffusion LoRA for {model_name_or_path}", kwargs, hf_token=hf_token
        )

    # Text model fine-tuning
    if not data_name:
        return {"status": "error", "error": "data_name is required for text model fine-tuning."}
    if not output_path:
        output_path = _make_output_path(Command.FINETUNE, model_name_or_path)
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
    return await _start_job(
        Command.FINETUNE, f"Finetune {model_name_or_path} ({method}) on {data_name}", kwargs, hf_token=hf_token
    )


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
        output_path = _make_output_path(Command.CAPTURE_ONNX_GRAPH, model_name_or_path)

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
    return await _start_job(
        Command.CAPTURE_ONNX_GRAPH, f"Capture ONNX from {model_name_or_path}", kwargs, hf_token=hf_token
    )


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
        output_path = _make_output_path(Command.BENCHMARK, model_name_or_path)

    kwargs = _build_kwargs(
        model_name_or_path=model_name_or_path,
        tasks=tasks,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        limit=limit,
        output_path=output_path,
    )
    return await _start_job(
        Command.BENCHMARK, f"Benchmark {model_name_or_path} on {', '.join(tasks)}", kwargs, hf_token=hf_token
    )


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
            if delete_all or (names and d.name in names) or (prefix and d.name.startswith(prefix + "_")):
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
    _known_prefixes = [
        Command.DIFFUSION_LORA,
        Command.CAPTURE_ONNX_GRAPH,
        Command.OPTIMIZE,
        Command.QUANTIZE,
        Command.FINETUNE,
        Command.BENCHMARK,
    ]

    entries = []
    for d in sorted(OUTPUT_BASE.iterdir(), key=lambda p: p.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        if prefix and not d.name.startswith(prefix):
            continue

        entry = {"name": d.name, "path": str(d)}

        for op in _known_prefixes:
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
