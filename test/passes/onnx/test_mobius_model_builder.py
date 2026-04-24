# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Unit tests for the MobiusModelBuilder Olive pass."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec, Device
from olive.hardware.constants import ExecutionProvider
from olive.model import HfModelHandler, ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.mobius_model_builder import MobiusModelBuilder


@pytest.fixture(autouse=True, scope="module")
def _stub_mobius_module():
    """Stub the optional mobius package into sys.modules for the duration of this module.

    patch("mobius.build") resolves the module via sys.modules, so it works correctly
    even in environments where mobius-ai is not installed (e.g. Olive CI).
    The stub is only injected when mobius is absent; if the real package is installed,
    this fixture is a no-op.
    """
    if "mobius" in sys.modules:
        yield
        return
    fake = types.ModuleType("mobius")
    fake.build = None  # overridden per-test by patch("mobius.build")
    sys.modules["mobius"] = fake
    yield
    sys.modules.pop("mobius", None)


@pytest.fixture(autouse=True)
def mock_hf_config():
    """Prevent HfModelHandler.__init__ from making network calls to resolve model configs."""
    mock_cfg = MagicMock()
    mock_cfg.to_dict.return_value = {}
    with (
        patch.object(HfModelHandler, "get_hf_model_config", return_value=mock_cfg),
        patch.object(HfModelHandler, "get_load_kwargs", return_value={}),
    ):
        yield


def _make_hf_model(model_path: str, load_kwargs: dict | None = None) -> HfModelHandler:
    model = HfModelHandler(model_path=model_path)
    if load_kwargs:
        # Patch get_load_kwargs on the instance to return the given kwargs.
        model.get_load_kwargs = lambda: load_kwargs
    return model


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
    # Patch mobius.build directly — lazy import inside _run_for_config means
    # patching the module attribute, not the local binding.
    # Also patch _write_genai_config since the default runtime is ort-genai.
    return _combine_patches(
        patch("mobius.build", return_value=pkg),
        patch.object(MobiusModelBuilder, "_write_genai_config"),
    )


class _combine_patches:
    """Combine multiple patch context managers into one."""

    def __init__(self, *patches):
        self._patches = patches
        self._mocks = []

    def __enter__(self):
        self._mocks = [p.__enter__() for p in self._patches]
        return self._mocks[0]  # return the build mock

    def __exit__(self, *args):
        for p in reversed(self._patches):
            p.__exit__(*args)


# ---------------------------------------------------------------------------
# Configuration tests
# ---------------------------------------------------------------------------


def test_default_config_params():
    """MobiusModelBuilder must declare precision and execution_provider, and must not declare trust_remote_code."""
    accelerator_spec = AcceleratorSpec(
        accelerator_type=Device.CPU, execution_provider=ExecutionProvider.CPUExecutionProvider
    )
    config = MobiusModelBuilder._default_config(accelerator_spec)  # pylint: disable=protected-access
    assert "precision" in config
    assert "execution_provider" in config
    assert "trust_remote_code" not in config


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


# ---------------------------------------------------------------------------
# Output validation tests
# ---------------------------------------------------------------------------


def test_missing_output_file_raises_runtime_error(tmp_path):
    """RuntimeError must be raised if pkg.save() does not produce model.onnx."""
    out = tmp_path / "out"
    # _fake_pkg normally writes the file; use a pkg whose save() does nothing.
    pkg = MagicMock()
    pkg.keys.return_value = ["model"]
    pkg.__iter__ = MagicMock(return_value=iter(["model"]))
    pkg.save.return_value = None  # save() succeeds but writes nothing

    with _patch_build(pkg), pytest.raises(RuntimeError, match="expected output file not found"):
        _make_pass().run(_make_hf_model("org/model"), out)


def test_missing_component_file_raises_runtime_error(tmp_path):
    """RuntimeError for multi-component if any component's model.onnx is missing."""
    out = tmp_path / "out"
    keys = ["model", "vision", "embedding"]
    pkg = MagicMock()
    pkg.keys.return_value = keys
    pkg.__iter__ = MagicMock(return_value=iter(keys))

    # save() only creates 'model' component, skips 'vision' and 'embedding'
    def _partial_save(directory: str, **_kwargs):
        d = Path(directory) / "model"
        d.mkdir(parents=True)
        (d / "model.onnx").write_text("dummy")

    pkg.save.side_effect = _partial_save

    with _patch_build(pkg), pytest.raises(RuntimeError, match="expected output file not found"):
        _make_pass().run(_make_hf_model("org/vlm"), out)


# ---------------------------------------------------------------------------
# Security / trust_remote_code tests
# ---------------------------------------------------------------------------


def test_trust_remote_code_warning_logged(tmp_path):
    """trust_remote_code=True on the model must emit a warning about trusted model sources."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)
    p = create_pass_from_dict(
        MobiusModelBuilder,
        {"precision": "fp32"},
        disable_search=True,
        accelerator_spec=AcceleratorSpec(
            accelerator_type=Device.CPU, execution_provider=ExecutionProvider.CPUExecutionProvider
        ),
    )
    with (
        _patch_build(pkg),
        patch("olive.passes.onnx.mobius_model_builder.logger") as mock_logger,
    ):
        p.run(_make_hf_model("org/model", load_kwargs={"trust_remote_code": True}), out)

    warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert any("trust_remote_code" in msg for msg in warning_messages)


def test_no_warning_when_trust_remote_code_false(tmp_path):
    """No trust_remote_code warning must be emitted when the model does not set trust_remote_code."""
    out = tmp_path / "out"
    pkg = _fake_pkg(["model"], out)
    with (
        _patch_build(pkg),
        patch("olive.passes.onnx.mobius_model_builder.logger") as mock_logger,
    ):
        _make_pass().run(_make_hf_model("org/model"), out)

    warning_messages = [call.args[0] for call in mock_logger.warning.call_args_list]
    assert not any("trust_remote_code" in msg for msg in warning_messages)
