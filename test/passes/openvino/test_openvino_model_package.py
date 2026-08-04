# -------------------------------------------------------------------------
# Copyright (c) Intel Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.context_binary import registered_ep_library
from olive.passes.openvino.model_package import OpenVINOModelPackage


def _make_model(path: Path, external_location: str = "weights.data", dim: int = 8):
    """Write a tiny MatMul ONNX whose single weight is external at `external_location`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = (path.parent / external_location)
    weights.parent.mkdir(parents=True, exist_ok=True)
    data = np.random.rand(dim, dim).astype(np.float32)
    weights.write_bytes(data.tobytes())

    w = TensorProto()
    w.name = "W"
    w.data_type = TensorProto.FLOAT
    w.dims.extend([dim, dim])
    w.data_location = TensorProto.EXTERNAL
    for k, v in (("location", external_location), ("offset", "0"), ("length", str(data.nbytes))):
        entry = w.external_data.add()
        entry.key, entry.value = k, v

    graph = helper.make_graph(
        [helper.make_node("MatMul", ["input", "W"], ["output"])],
        "g",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, dim])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, dim])],
        initializer=[w],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 10
    onnx.save(model, str(path))
    return path


def _pass(tmp_path, config):
    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="OpenVINOExecutionProvider")
    return create_pass_from_dict(OpenVINOModelPackage, config, disable_search=True, accelerator_spec=accelerator_spec)


# ─── _external_data_files ────────────────────────────────────────────────────


def test_external_data_files_preserves_relative_location(tmp_path):
    model = _make_model(tmp_path / "model.onnx", external_location="weights/model.data")
    files = OpenVINOModelPackage._external_data_files(model)
    assert [loc for loc, _ in files] == ["weights/model.data"]
    assert files[0][1] == tmp_path / "weights" / "model.data"


def test_external_data_files_rejects_missing_file(tmp_path):
    model = _make_model(tmp_path / "model.onnx")
    (tmp_path / "weights.data").unlink()
    with pytest.raises(FileNotFoundError):
        OpenVINOModelPackage._external_data_files(model)


# ─── _add_shared_asset ───────────────────────────────────────────────────────


def test_add_shared_asset_preserves_nested_layout(tmp_path):
    model = _make_model(tmp_path / "model.onnx", external_location="weights/model.data")
    package_dir = tmp_path / "pkg"
    uri = OpenVINOModelPackage._add_shared_asset(model, package_dir)

    assert uri.startswith("sha256:")
    asset_dir = package_dir / "shared_assets" / f"sha256-{uri.split(':')[1]}"
    # nested external-data location is preserved, not flattened to the basename
    assert (asset_dir / "model.onnx").is_file()
    assert (asset_dir / "weights" / "model.data").is_file()
    assert not (asset_dir / "model.data").exists()


def test_add_asset_entry_detects_collision():
    entries = {}
    OpenVINOModelPackage._add_asset_entry(entries, "w.data", Path("a/w.data"))
    # same file + location is idempotent
    OpenVINOModelPackage._add_asset_entry(entries, "w.data", Path("a/w.data"))
    # distinct file mapping to the same location is rejected
    with pytest.raises(ValueError, match="same package location"):
        OpenVINOModelPackage._add_asset_entry(entries, "w.data", Path("b/w.data"))


# ─── failure behavior ────────────────────────────────────────────────────────


def test_fails_when_requested_target_fails(tmp_path):
    """Default (allow_partial_package=False): any failed target fails the whole pass."""
    model = ONNXModelHandler(model_path=str(_make_model(tmp_path / "model.onnx")))
    p = _pass(tmp_path, {"package_name": "m", "target_devices": ["643E", "B03E"], "create_shared_asset": False})

    targets = [{"device_id_hex": "643E", "npu_platform": "4000"}, {"device_id_hex": "B03E", "npu_platform": "5010"}]

    def compile_side_effect(config, model_path, output_ctx, target):
        if target["device_id_hex"] == "B03E":
            raise RuntimeError("compiler blew up")
        Path(output_ctx).write_text("ctx")

    with (
        patch.object(OpenVINOModelPackage, "_discover_targets", return_value=targets),
        patch.object(OpenVINOModelPackage, "_compile_variant", side_effect=compile_side_effect),
        patch.object(OpenVINOModelPackage, "_read_compat", return_value=""),
    ):
        with pytest.raises(RuntimeError, match="B03E"):
            p.run(model, str(tmp_path / "out"))


def test_partial_package_when_allowed(tmp_path):
    """allow_partial_package=True: failed target is skipped, package built from the rest."""
    model = ONNXModelHandler(model_path=str(_make_model(tmp_path / "model.onnx")))
    p = _pass(
        tmp_path,
        {
            "package_name": "m",
            "target_devices": ["643E", "B03E"],
            "create_shared_asset": False,
            "allow_partial_package": True,
        },
    )

    targets = [{"device_id_hex": "643E", "npu_platform": "4000"}, {"device_id_hex": "B03E", "npu_platform": "5010"}]

    def compile_side_effect(config, model_path, output_ctx, target):
        if target["device_id_hex"] == "B03E":
            raise RuntimeError("compiler blew up")
        Path(output_ctx).write_text("ctx")

    with (
        patch.object(OpenVINOModelPackage, "_discover_targets", return_value=targets),
        patch.object(OpenVINOModelPackage, "_compile_variant", side_effect=compile_side_effect),
        patch.object(OpenVINOModelPackage, "_read_compat", return_value=""),
    ):
        p.run(model, str(tmp_path / "out"))

    manifest = json.loads((tmp_path / "out" / "m.ortpackage" / "manifest.json").read_text())
    variants = manifest["components"]["m"]["variants"]
    assert set(variants) == {"m.npu_643E"}


# ─── non-empty package dir guard ─────────────────────────────────────────────


def test_refuses_non_empty_package_dir(tmp_path):
    model = ONNXModelHandler(model_path=str(_make_model(tmp_path / "model.onnx")))
    stale = tmp_path / "out" / "m.ortpackage"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("old")
    p = _pass(tmp_path, {"package_name": "m", "target_devices": ["643E"], "create_shared_asset": False})

    with (
        patch.object(OpenVINOModelPackage, "_discover_targets", return_value=[{"device_id_hex": "643E", "npu_platform": "4000"}]),
        pytest.raises(FileExistsError, match="not empty"),
    ):
        p.run(model, str(tmp_path / "out"))


# ─── EP registration ownership ───────────────────────────────────────────────


def test_registered_ep_library_unregisters_when_owned():
    ort = MagicMock()
    with registered_ep_library(ort, "name", "lib"):
        pass
    ort.register_execution_provider_library.assert_called_once()
    ort.unregister_execution_provider_library.assert_called_once()


def test_registered_ep_library_skips_unregister_when_not_owned():
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("EP library already registered")
    with registered_ep_library(ort, "name", "lib"):
        pass
    ort.unregister_execution_provider_library.assert_not_called()


def test_registered_ep_library_propagates_real_error():
    ort = MagicMock()
    ort.register_execution_provider_library.side_effect = RuntimeError("bad path")
    with pytest.raises(RuntimeError, match="bad path"):
        with registered_ep_library(ort, "name", "lib"):
            pass
    ort.unregister_execution_provider_library.assert_not_called()


# ─── package_only manifest shape ─────────────────────────────────────────────


def test_package_only_manifest_shape(tmp_path):
    """package_only assembles a stable manifest from pre-compiled ctx models, no compile."""
    model = ONNXModelHandler(model_path=str(_make_model(tmp_path / "model.onnx")))

    # pre-compiled ctx models (+ .bin sidecar) per device
    compiled = {}
    for dev in ("643E", "B03E"):
        ctx_dir = tmp_path / "compiled" / dev
        ctx_dir.mkdir(parents=True)
        (ctx_dir / "model_ctx.onnx").write_text("ctx")
        (ctx_dir / "model_ctx_OpenVINOExecutionProvider.bin").write_text("bin")
        compiled[dev] = str(ctx_dir / "model_ctx.onnx")

    p = _pass(
        tmp_path,
        {
            "package_name": "m",
            "package_only": True,
            "shared_asset_model_path": str(tmp_path / "model.onnx"),
            "compiled_model_paths": compiled,
            "target_devices": ["643E", "B03E"],
        },
    )

    # the dummy ctx files are not real ONNX, so stub the ONNX-parsing helpers
    with (
        patch.object(OpenVINOModelPackage, "_read_compat", return_value="compat-643E"),
        patch.object(OpenVINOModelPackage, "_compiled_context_auxiliary_files") as mock_aux,
    ):
        mock_aux.side_effect = lambda ctx: [
            (p.name, p) for p in ctx.parent.glob(f"{ctx.stem}*.bin")
        ]
        p.run(model, str(tmp_path / "out"))

    pkg = tmp_path / "out" / "m.ortpackage"
    manifest = json.loads((pkg / "manifest.json").read_text())

    assert manifest["schema_version"] == "1.0"
    assert set(manifest["components"]["m"]["variants"]) == {"m.npu_643E", "m.npu_B03E"}
    variant = manifest["components"]["m"]["variants"]["m.npu_643E"]
    assert variant["ep"] == "OpenVINOExecutionProvider"
    assert variant["variant_directory"] == "m/m.npu_643E"
    assert variant["executor_info"]["ort"]["model_file"] == "model_ctx.onnx"
    # shared-asset uri is injected into each variant's session_options
    assert variant["executor_info"]["ort"]["session_options"][
        "session.model_external_initializers_file_folder_path"
    ].startswith("sha256:")
    # ctx + bin copied into the variant dir; shared asset holds the original model + weights
    assert (pkg / "m" / "m.npu_643E" / "model_ctx.onnx").is_file()
    assert (pkg / "m" / "m.npu_643E" / "model_ctx_OpenVINOExecutionProvider.bin").is_file()
    assert list((pkg / "shared_assets").iterdir())
