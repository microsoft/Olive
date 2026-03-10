# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import os
import sys
import traceback
from pathlib import Path


def _get_model_size_mb(model_path: str) -> float | None:
    """Calculate total size of model files in MB."""
    try:
        p = Path(os.path.normpath(model_path))
        if p.is_file():
            return round(p.stat().st_size / (1024 * 1024), 2)
        elif p.is_dir():
            model_exts = (".onnx", ".pt", ".bin", ".safetensors", ".onnx_data", ".pb", ".weight")
            total = 0
            for dirpath, _, filenames in os.walk(p):
                for f in filenames:
                    if any(f.endswith(ext) for ext in model_exts):
                        total += Path(dirpath, f).stat().st_size
            return round(total / (1024 * 1024), 2) if total > 0 else None
        return None
    except Exception:
        return None


def serialize_workflow_output(result):
    """Convert a WorkflowOutput (or None) into a JSON-serializable dict."""
    if result is None:
        return {"status": "success", "output_models": []}

    output = {
        "status": "success",
        "device": result.from_device(),
        "execution_provider": result.from_execution_provider(),
        "output_models": [],
    }

    for m in result.get_output_models():
        model_info = {
            "model_path": m.model_path,
            "model_id": m.model_id,
            "model_type": m.model_type,
            "metrics": m.metrics_value,
            "inference_config": m.get_inference_config() or None,
        }
        if m.model_path:
            model_info["file_size_mb"] = _get_model_size_mb(m.model_path)
        output["output_models"].append(model_info)

    best = result.get_best_candidate()
    if best:
        output["best_model"] = {
            "model_path": best.model_path,
            "model_id": best.model_id,
            "model_type": best.model_type,
            "metrics": best.metrics_value,
            "inference_config": best.get_inference_config() or None,
        }
        if best.model_path:
            output["best_model"]["file_size_mb"] = _get_model_size_mb(best.model_path)

    # Pass execution summary from footprint
    try:
        run_history = result.footprint.summarize_run_history()
        pass_summary = []
        total_duration = 0.0
        for rh in run_history:
            entry = {
                "model_id": rh.model_id,
                "parent_model_id": rh.parent_model_id,
                "from_pass": rh.from_pass,
                "duration_sec": round(rh.duration_sec, 2) if rh.duration_sec is not None else None,
            }
            if rh.duration_sec is not None:
                total_duration += rh.duration_sec
            pass_summary.append(entry)
        output["pass_summary"] = pass_summary
        output["total_duration_seconds"] = round(total_duration, 2)
    except Exception:
        # Optional enrichment — log but don't break serialization
        traceback.print_exc(file=sys.stderr)

    # Input model info for before/after comparison
    try:
        input_metrics = result.get_input_model_metrics()
        if input_metrics:
            output["input_model_metrics"] = input_metrics

        input_model_id = result.footprint.input_model_id
        if input_model_id:
            input_node = result.footprint.nodes.get(input_model_id)
            if input_node and input_node.model_config_data:
                input_path = input_node.model_config_data.get("config", {}).get("model_path")
                if input_path:
                    input_size = _get_model_size_mb(input_path)
                    if input_size is not None:
                        output["input_model_size_mb"] = input_size
    except Exception:
        # Optional enrichment — log but don't break serialization
        traceback.print_exc(file=sys.stderr)

    return output


# ---------------------------------------------------------------------------
# explore_passes — introspect Olive pass registry
# ---------------------------------------------------------------------------


def _serialize_type(type_annotation) -> str:
    """Convert a type annotation to a readable string."""
    if type_annotation is None:
        return "any"
    name = getattr(type_annotation, "__name__", None)
    if name:
        return name
    return str(type_annotation).replace("typing.", "")


def _serialize_default(value):
    """Convert a default value to something JSON-serializable."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_serialize_default(v) for v in value]
    if isinstance(value, dict):
        return {k: _serialize_default(v) for k, v in value.items()}
    return str(value)


def _handle_explore_passes(kwargs):
    """List or inspect Olive passes."""
    from olive.package_config import OlivePackageConfig
    from olive.passes.olive_pass import Pass

    olive_config = OlivePackageConfig.load_default_config()
    pass_registry = Pass.registry

    pass_name = kwargs.get("pass_name")
    provider = kwargs.get("provider")
    precision = kwargs.get("precision")
    accelerator = kwargs.get("accelerator")

    # --- Detail mode ---
    if pass_name:
        canonical_name = None
        module_config = None
        for key in olive_config.passes:
            if key.lower() == pass_name.lower():
                canonical_name = key
                module_config = olive_config.passes[key]
                break
        if not module_config:
            return {"error": f"Pass '{pass_name}' not found. Available: {sorted(olive_config.passes.keys())}"}

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

        # Try to get full parameter schema from pass class
        cls = pass_registry.get(pass_name.lower())
        if not cls and module_config.module_path:
            import importlib

            parts = module_config.module_path.rsplit(".", 1)
            if len(parts) == 2:
                try:
                    mod = importlib.import_module(parts[0])
                    cls = getattr(mod, parts[1], None)
                except Exception:
                    # Dynamic import may fail if deps are missing — skip schema
                    cls = None

        if cls:
            try:
                from olive.hardware import AcceleratorSpec, Device
                from olive.hardware.constants import ExecutionProvider

                spec = AcceleratorSpec(
                    accelerator_type=Device.CPU,
                    execution_provider=ExecutionProvider.CPUExecutionProvider,
                )
                default_config = cls._default_config(spec)  # pylint: disable=protected-access
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

    # --- List mode ---
    passes = []
    for name, mc in sorted(olive_config.passes.items()):
        if provider and "*" not in mc.supported_providers and provider not in mc.supported_providers:
            continue
        if precision and "*" not in mc.supported_precisions and precision not in mc.supported_precisions:
            continue
        if accelerator and "*" not in mc.supported_accelerators and accelerator not in mc.supported_accelerators:
            continue
        passes.append(
            {
                "name": name,
                "providers": list(mc.supported_providers),
                "accelerators": list(mc.supported_accelerators),
                "precisions": list(mc.supported_precisions),
                "algorithms": list(mc.supported_algorithms or []),
                "dataset": mc.dataset,
            }
        )

    return {"passes": passes, "total": len(passes)}


def _handle_validate_config(kwargs):
    """Validate an Olive workflow config."""
    from olive.workflows.run.config import RunConfig

    config = kwargs.get("config")
    try:
        RunConfig.parse_file_or_obj(config)
        return {"valid": True, "message": "Config is valid."}
    except Exception as e:
        errors = []
        if hasattr(e, "errors"):
            for err in e.errors():
                loc = ".".join(str(x) for x in err.get("loc", []))
                errors.append({"location": loc, "message": err.get("msg", ""), "type": err.get("type", "")})
        return {"valid": False, "errors": errors if errors else [{"message": str(e)}]}


def _handle_run_config(kwargs):
    """Run an Olive workflow config."""
    from olive.cli.api import run as olive_run

    config_file = kwargs.pop("config_file")
    kwargs.setdefault("log_level", 1)
    workflow_output = olive_run(run_config=config_file, **kwargs)
    return serialize_workflow_output(workflow_output)


def main():
    # Redirect stdout to stderr so olive's internal prints don't pollute our JSON output.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        command = sys.argv[1]
        kwargs = json.loads(sys.argv[2])

        # Special commands (no olive.cli.api dispatch)
        if command == "explore_passes":
            result = _handle_explore_passes(kwargs)
        elif command == "validate_config":
            result = _handle_validate_config(kwargs)
        elif command == "run_config":
            result = _handle_run_config(kwargs)
        else:
            from olive.cli.api import (
                benchmark,
                capture_onnx_graph,
                diffusion_lora,
                finetune,
                optimize,
                quantize,
            )

            dispatch = {
                "optimize": optimize,
                "quantize": quantize,
                "finetune": finetune,
                "capture_onnx_graph": capture_onnx_graph,
                "benchmark": benchmark,
                "diffusion_lora": diffusion_lora,
            }

            func = dispatch.get(command)
            if func is None:
                result = {"status": "error", "error": f"Unknown command: {command}"}
            else:
                # Default log_level to 1 (INFO) so olive outputs progress to stderr.
                # Olive default is 3 (ERROR) which hides almost everything.
                # 0=DEBUG, 1=INFO, 2=WARNING, 3=ERROR, 4=CRITICAL
                kwargs.setdefault("log_level", 1)
                workflow_output = func(**kwargs)
                result = serialize_workflow_output(workflow_output)

    except Exception:
        result = {"status": "error", "error": traceback.format_exc()[-3000:]}

    # Write JSON result to original stdout
    original_stdout.write(json.dumps(result))
    original_stdout.flush()


if __name__ == "__main__":
    main()
