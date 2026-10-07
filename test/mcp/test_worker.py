# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib.util
from pathlib import Path

# pylint: disable=protected-access


def _load_worker_module():
    worker_path = Path(__file__).parents[2] / "mcp" / "src" / "olive_mcp" / "worker.py"
    spec = importlib.util.spec_from_file_location("olive_mcp_worker_test", worker_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MCP worker from {worker_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


worker = _load_worker_module()


def test_serialize_workflow_output_handles_build_mapping():
    result = worker.serialize_workflow_output({"first": None, "second": None})

    assert result == {
        "status": "success",
        "builds": {
            "first": {"status": "success", "output_models": []},
            "second": {"status": "success", "output_models": []},
        },
    }


def test_validate_config_accepts_multi_build_config():
    config = {
        "input_model": {"type": "ONNXModel", "model_path": "model.onnx"},
        "passes": {
            "convert": {"type": "OnnxConversion"},
            "tune": {"type": "OrtSessionParamsTuning"},
        },
        "evaluate_input_model": False,
        "builds": {
            "convert": {"pipeline": ["convert"], "output_dir": "out/convert"},
            "tune": {"pipeline": ["tune"], "output_dir": "out/tune"},
        },
    }

    assert worker._handle_validate_config({"config": config}) == {"valid": True, "message": "Config is valid."}
