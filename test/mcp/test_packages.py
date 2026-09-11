# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import importlib
import importlib.util
import sys
import types
from pathlib import Path

# pylint: disable=protected-access


def _load_packages_module(monkeypatch):
    constants = types.ModuleType("olive_mcp.constants")
    constants.PROVIDER_TO_EXTRAS = {}
    constants.PROVIDER_TO_GENAI = {"CPUExecutionProvider": "onnxruntime-genai"}
    constants.Command = types.SimpleNamespace(
        OPTIMIZE="optimize",
        QUANTIZE="quantize",
        FINETUNE="finetune",
        CAPTURE_ONNX_GRAPH="capture-onnx-graph",
        BENCHMARK="benchmark",
        DIFFUSION_LORA="diffusion-lora",
    )
    monkeypatch.setitem(sys.modules, "olive_mcp.constants", constants)

    packages_path = Path(__file__).parents[2] / "mcp" / "src" / "olive_mcp" / "packages.py"
    spec = importlib.util.spec_from_file_location("olive_mcp_packages_test", packages_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load MCP packages from {packages_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_resolve_optimize_packages_includes_mobius(monkeypatch):
    packages = _load_packages_module(monkeypatch)

    resolved = packages._resolve_packages(packages.Command.OPTIMIZE, exporter="mobius")

    assert "olive-ai[cpu]" in resolved
    assert "mobius-onnx" in resolved


def test_resolve_optimize_packages_keeps_default_model_builder(monkeypatch):
    packages = _load_packages_module(monkeypatch)

    resolved = packages._resolve_packages(packages.Command.OPTIMIZE)

    assert "olive-ai[cpu]" in resolved
    assert "onnxruntime-genai" in resolved
    assert "mobius-onnx" not in resolved
