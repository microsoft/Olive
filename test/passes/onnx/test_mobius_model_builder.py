# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the MobiusModelBuilder Olive pass."""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mobius_model_builder import MobiusModelBuilder


@pytest.fixture(autouse=True)
def mock_hf_config():
    """Prevent HfModelHandler.__init__ from making network calls to resolve model configs."""
    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}
    with patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg):
        yield


def _make_hf_model(model_path: str) -> HfModelHandler:
    return HfModelHandler(model_path=model_path)


def _make_pass(ep: str = ExecutionProvider.CPUExecutionProvider) -> MobiusModelBuilder:
    accelerator_spec = AcceleratorSpec(accelerator_type=Device.CPU, execution_provider=ep)
    return create_pass_from_dict(
        MobiusModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )


def _fake_pkg(keys: list[str], _output_dir: Path) -> MagicMock:
    """Create a fake ModelPackage that writes dummy .onnx files when .save() is called."""

    def _save(directory: str, **_kwargs):
        out = Path(directory)
        if len(keys) == 1:
            # Single-component: saved as <dir>/model.onnx
            (out / "model.onnx").write_text("dummy")
        else:
            # Multi-component: saved as <dir>/<key>/model.onnx
            for k in keys:
                (out / k).mkdir(parents=True, exist_ok=True)
                (out / k / "model.onnx").write_text("dummy")

    pkg = MagicMock()
    pkg.keys.return_value = keys
    pkg.__iter__ = MagicMock(return_value=iter(keys))
    pkg.items.return_value = [(k, MagicMock()) for k in keys]
    pkg.save.side_effect = _save
    return pkg


def _patch_build(pkg: MagicMock):
    """Patch mobius.build (the real import target used by _run_for_config)."""
    return patch("mobius.build", return_value=pkg)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_default_config_params():
    """MobiusModelBuilder must declare precision, execution_provider, trust_remote_code."""
    accelerator_spec = AcceleratorSpec(
        accelerator_type=Device.CPU, execution_provider=ExecutionProvider.CPUExecutionProvider
    )
    config = MobiusModelBuilder._default_config(accelerator_spec)
    assert "precision" in config
    assert "execution_provider" in config
    assert "trust_remote_code" in config


def test_is_not_accelerator_agnostic():
    """Pass must be EP-specific because it chooses fused ops based on the EP."""
    accelerator_spec = AcceleratorSpec(
        accelerator_type=Device.CPU, execution_provider=ExecutionProvider.CPUExecutionProvider
    )
    assert MobiusModelBuilder.is_accelerator_agnostic(accelerator_spec) is False


def test_ep_map_covers_common_providers():
    assert ExecutionProvider.CPUExecutionProvider in MobiusModelBuilder.EP_MAP
    assert ExecutionProvider.CUDAExecutionProvider in MobiusModelBuilder.EP_MAP
    assert ExecutionProvider.DmlExecutionProvider in MobiusModelBuilder.EP_MAP
    assert ExecutionProvider.WebGpuExecutionProvider in MobiusModelBuilder.EP_MAP
    assert MobiusModelBuilder.EP_MAP[ExecutionProvider.CPUExecutionProvider] == "cpu"
    assert MobiusModelBuilder.EP_MAP[ExecutionProvider.CUDAExecutionProvider] == "cuda"
    assert MobiusModelBuilder.EP_MAP[ExecutionProvider.DmlExecutionProvider] == "dml"
    assert MobiusModelBuilder.EP_MAP[ExecutionProvider.WebGpuExecutionProvider] == "webgpu"


# ---------------------------------------------------------------------------
# Single-component model tests
# ---------------------------------------------------------------------------


def test_single_component_returns_onnx_handler(tmp_path):
    """Single-component package (e.g. LLM) → ONNXModelHandler."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)

    with _patch_build(pkg) as mock_build:
        p = _make_pass()
        result = p.run(_make_hf_model("meta-llama/Llama-3-8B"), out)

    assert isinstance(result, ONNXModelHandler)
    assert not isinstance(result, CompositeModelHandler)
    assert Path(result.model_path).exists()
    mock_build.assert_called_once()
    call_kwargs = mock_build.call_args.kwargs
    assert call_kwargs["execution_provider"] == "cpu"
    assert call_kwargs["dtype"] == "f32"


def test_model_onnx_exists_after_run(tmp_path):
    """The saved model.onnx file must exist on disk."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)

    with _patch_build(pkg):
        p = _make_pass()
        result = p.run(_make_hf_model("org/model"), out)

    # ONNXModelHandler.model_path already points to the .onnx file
    assert Path(result.model_path).exists()


# ---------------------------------------------------------------------------
# Multi-component model tests
# ---------------------------------------------------------------------------


def test_multi_component_returns_composite_handler(tmp_path):
    """Multi-component package (VLM) → CompositeModelHandler with one component per key."""
    out = tmp_path / "out"
    keys = ["model", "vision", "embedding"]
    pkg = _fake_pkg(keys, out)

    with _patch_build(pkg):
        p = _make_pass()
        result = p.run(_make_hf_model("microsoft/phi-4-vision"), out)

    assert isinstance(result, CompositeModelHandler)
    assert result.model_component_names == keys
    components = list(result.model_components)
    assert len(components) == 3
    for comp in components:
        assert isinstance(comp, ONNXModelHandler)


# ---------------------------------------------------------------------------
# EP auto-detection tests
# ---------------------------------------------------------------------------


def test_ep_auto_detected_from_accelerator(tmp_path):
    """When execution_provider config is None, use the Olive accelerator EP."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)

    accelerator_spec = AcceleratorSpec(
        accelerator_type=Device.GPU, execution_provider=ExecutionProvider.CUDAExecutionProvider
    )
    p = create_pass_from_dict(
        MobiusModelBuilder,
        {"precision": "fp16"},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    with _patch_build(pkg) as mock_build:
        p.run(_make_hf_model("org/model"), out)

    call_kwargs = mock_build.call_args.kwargs
    assert call_kwargs["execution_provider"] == "cuda"
    assert call_kwargs["dtype"] == "f16"


def test_ep_override_from_config(tmp_path):
    """Explicit execution_provider in config overrides the accelerator EP."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)

    accelerator_spec = AcceleratorSpec(
        accelerator_type=Device.GPU, execution_provider=ExecutionProvider.CUDAExecutionProvider
    )
    p = create_pass_from_dict(
        MobiusModelBuilder,
        {"precision": "fp32", "execution_provider": "webgpu"},
        disable_search=True,
        accelerator_spec=accelerator_spec,
    )

    with _patch_build(pkg) as mock_build:
        p.run(_make_hf_model("org/model"), out)

    assert mock_build.call_args.kwargs["execution_provider"] == "webgpu"


# ---------------------------------------------------------------------------
# Input validation tests
# ---------------------------------------------------------------------------


def test_non_hf_model_raises(tmp_path):
    """Passing a non-HfModelHandler must raise ValueError."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "model.onnx").write_bytes(b"")

    onnx_model = ONNXModelHandler(model_path=str(out), onnx_file_name="model.onnx")
    p = _make_pass()
    with pytest.raises(ValueError, match="HfModelHandler"):
        p.run(onnx_model, tmp_path / "result")


def test_import_error_raised_when_mobius_missing(tmp_path):
    """ImportError must surface clearly when mobius is not installed."""
    p = _make_pass()
    with patch.dict(sys.modules, {"mobius": None}), pytest.raises(ImportError, match="mobius"):
        p.run(_make_hf_model("org/model"), tmp_path / "out")
