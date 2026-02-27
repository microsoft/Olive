"""Olive MCP Worker - runs olive.cli.api in an isolated venv.

This script is invoked by the MCP server in a task-specific virtual environment.
It calls the Olive Python API and returns structured JSON results via stdout.

Usage: python worker.py <command> <json_kwargs>
"""

import json
import sys
import traceback


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
        output["output_models"].append({
            "model_path": m.model_path,
            "model_id": m.model_id,
            "model_type": m.model_type,
            "metrics": m.metrics_value,
            "inference_config": m.get_inference_config() or None,
        })

    best = result.get_best_candidate()
    if best:
        output["best_model"] = {
            "model_path": best.model_path,
            "model_id": best.model_id,
            "model_type": best.model_type,
            "metrics": best.metrics_value,
            "inference_config": best.get_inference_config() or None,
        }

    return output


def main():
    # Redirect stdout to stderr so olive's internal prints don't pollute our JSON output.
    original_stdout = sys.stdout
    sys.stdout = sys.stderr

    try:
        command = sys.argv[1]
        kwargs = json.loads(sys.argv[2])

        from olive.cli.api import (
            benchmark,
            capture_onnx_graph,
            diffusion_lora,
            finetune,
            optimize,
            quantize,
            tune_session_params,
        )

        dispatch = {
            "optimize": optimize,
            "quantize": quantize,
            "finetune": finetune,
            "capture_onnx_graph": capture_onnx_graph,
            "benchmark": benchmark,
            "diffusion_lora": diffusion_lora,
            "tune_session_params": tune_session_params,
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
