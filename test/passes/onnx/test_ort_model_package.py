# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_package import ORTModelPackage

_EP = "OpenVINOExecutionProvider"


def _make_model(path: Path, external_location: str = "weights.data", dim: int = 8):
    """Write a tiny MatMul ONNX whose single weight is external at `external_location`."""
    path.parent.mkdir(parents=True, exist_ok=True)
    weights = path.parent / external_location
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


def _make_ctx_model(directory: Path, name: str = "model_ctx.onnx", bin_name: str = "model_ctx_ep.bin") -> Path:
    """Write an EPContext ONNX model that references a sidecar .bin file."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / bin_name).write_bytes(b"bin")
    node = helper.make_node("EPContext", [], [], domain="com.microsoft", ep_cache_context=bin_name, embed_mode=0)
    model = helper.make_model(
        helper.make_graph([node], "ctx", [], []), opset_imports=[helper.make_opsetid("com.microsoft", 1)]
    )
    onnx.save(model, str(directory / name))
    return directory / name


def _composite(root: Path, variant_ids, source_model: Path = None):
    components = [
        ONNXModelHandler(model_path=str(_make_ctx_model(root / variant_id).parent), onnx_file_name="model_ctx.onnx")
        for variant_id in variant_ids
    ]
    attributes = {"source_model_path": str(source_model)} if source_model else None
    return CompositeModelHandler(components, list(variant_ids), model_path=str(root), model_attributes=attributes)


def _pass(config):
    accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider=_EP)
    return create_pass_from_dict(ORTModelPackage, config, disable_search=True, accelerator_spec=accelerator_spec)


@pytest.fixture
def native_model_package():
    import onnxruntime as ort

    if not hasattr(ort, "ModelPackage"):
        pytest.skip("Requires an ONNX Runtime build with ModelPackage Python bindings")
    return ort.ModelPackage


@pytest.fixture
def mock_model_package():
    # Exercise pass orchestration even with ORT builds that lack the authoring bindings.
    with patch("onnxruntime.__version__", "1.32.0"), patch("onnxruntime.ModelPackage", create=True) as constructor:
        package = constructor.return_value.__enter__.return_value
        package.add_shared_asset.return_value = "sha256:" + "a" * 64

        def commit(dest_root=None):
            if dest_root is not None:
                Path(dest_root).mkdir(parents=True)

        package.commit.side_effect = commit
        yield package


# ─── package assembly ────────────────────────────────────────────────────────


def test_composite_input_creates_variant_per_component(tmp_path, native_model_package):
    source = _make_model(tmp_path / "src" / "model.onnx", external_location="weights/model.data")
    model = _composite(tmp_path / "compiled", ["643E", "B03E"], source_model=source)
    p = _pass({"package_name": "m", "session_options": {"ep.context_file_path": "model_ctx.onnx"}})

    with patch.object(ORTModelPackage, "_read_compat", return_value="compat-1"):
        result = p.run(model, str(tmp_path / "out"))

    pkg = tmp_path / "out" / "m.ortpackage"
    manifest = json.loads((pkg / "manifest.json").read_text())

    assert manifest["schema_version"] == "1.0"
    assert set(manifest["components"]["m"]["variants"]) == {"m.npu_643E", "m.npu_B03E"}
    variant = manifest["components"]["m"]["variants"]["m.npu_643E"]
    assert variant["ep"] == _EP
    assert variant["variant_directory"] == "m/m.npu_643E"
    assert variant["compatibility_string"] == "compat-1"
    assert variant["executor_info"]["ort"]["model_file"] == "model_ctx.onnx"
    session_options = variant["executor_info"]["ort"]["session_options"]
    assert session_options["ep.context_file_path"] == "model_ctx.onnx"
    # shared-asset uri is injected into each variant's session_options
    assert session_options["session.model_external_initializers_file_folder_path"].startswith("sha256:")

    # ctx + referenced bin copied into each variant dir; shared asset holds the original model + weights
    assert (pkg / "m" / "m.npu_643E" / "model_ctx.onnx").is_file()
    assert (pkg / "m" / "m.npu_643E" / "model_ctx_ep.bin").is_file()
    (shared_asset,) = list((pkg / "shared_assets").iterdir())
    assert (shared_asset / "model.onnx").is_file()
    assert (shared_asset / "weights" / "model.data").is_file()

    assert isinstance(result, CompositeModelHandler)
    assert result.model_component_names == ["m.npu_643E", "m.npu_B03E"]
    assert result.model_attributes["no_flatten"] is True


def test_single_model_input_uses_default_variant(tmp_path, mock_model_package):
    ctx = _make_ctx_model(tmp_path / "compiled")
    source = _make_model(tmp_path / "src" / "model.onnx")
    model = ONNXModelHandler(model_path=str(ctx.parent), onnx_file_name=ctx.name)
    p = _pass({"package_name": "m", "shared_asset_model_path": str(source)})

    with patch.object(ORTModelPackage, "_read_compat", return_value=""):
        p.run(model, str(tmp_path / "out"))

    assert mock_model_package.set_variant.call_args.args[:2] == ("m", "m.npu_default")


def test_package_uses_native_authoring_api(tmp_path, mock_model_package):
    source = _make_model(tmp_path / "src" / "model.onnx")
    model = _composite(tmp_path / "compiled", ["643E"], source_model=source)
    p = _pass({"package_name": "m"})
    with patch.object(ORTModelPackage, "_read_compat", return_value="compat"):
        result = p.run(model, str(tmp_path / "out"))

    mock_model_package.set_metadata.assert_called_once_with(name="m", version="1.0.0")
    mock_model_package.set_component_inline.assert_called_once_with("m", json.dumps({"variants": {}}))
    mock_model_package.add_shared_asset.assert_called_once()
    args = mock_model_package.set_variant.call_args.args
    assert args[:2] == ("m", "m.npu_643E")
    assert json.loads(args[2]) == {
        "variant_directory": "m/m.npu_643E",
        "ep": _EP,
        "device": "npu",
        "compatibility_string": "compat",
    }
    executor_args = mock_model_package.set_variant_executor_info_inline.call_args.args
    assert executor_args[:3] == ("m", "m.npu_643E", "ort")
    assert json.loads(executor_args[3]) == {
        "model_file": "model_ctx.onnx",
        "session_options": {"session.model_external_initializers_file_folder_path": "sha256:" + "a" * 64},
    }
    assert mock_model_package.commit.call_count == 2
    assert mock_model_package.commit.call_args.args == ()
    assert Path(result.model_path) == tmp_path / "out"
    assert (tmp_path / "out" / "m.ortpackage" / "m" / "m.npu_643E" / "model_ctx.onnx").is_file()
    assert not list((tmp_path / "out").glob(".ortpackage-*"))


def test_package_name_defaults_to_shared_asset_model_stem(tmp_path, mock_model_package):
    source = _make_model(tmp_path / "src" / "my_model.onnx")
    model = _composite(tmp_path / "compiled", ["643E"], source_model=source)
    with patch.object(ORTModelPackage, "_read_compat", return_value=""):
        _pass({}).run(model, str(tmp_path / "out"))

    assert mock_model_package.set_metadata.call_args.kwargs["name"] == "my_model"
    assert (tmp_path / "out" / "my_model.ortpackage").is_dir()


# ─── input validation ────────────────────────────────────────────────────────


def test_missing_native_authoring_api_fails_before_output(tmp_path):
    import onnxruntime as ort

    model = _composite(tmp_path / "compiled", ["643E"])
    p = _pass({"package_name": "m", "create_shared_asset": False})

    with patch.object(ort, "ModelPackage", create=True):
        del ort.ModelPackage
        with pytest.raises(RuntimeError, match="onnxruntime.ModelPackage"):
            p.run(model, str(tmp_path / "out"))
    assert not (tmp_path / "out").exists()


def test_create_shared_asset_requires_source_model(tmp_path, mock_model_package):
    model = _composite(tmp_path / "compiled", ["643E"])
    with pytest.raises(ValueError, match="source_model_path"):
        _pass({"package_name": "m"}).run(model, str(tmp_path / "out"))


def test_weightless_variants_require_weights_source_when_asset_disabled(tmp_path, mock_model_package):
    source = _make_model(tmp_path / "src" / "model.onnx")
    model = _composite(tmp_path / "compiled", ["643E"])
    p = _pass({"package_name": "m", "create_shared_asset": False, "shared_asset_model_path": str(source)})
    with pytest.raises(ValueError, match="requires session_options"):
        p.run(model, str(tmp_path / "out"))


def test_weightless_variants_check_source_model_attribute_when_asset_disabled(tmp_path, mock_model_package):
    source = _make_model(tmp_path / "src" / "model.onnx")
    model = _composite(tmp_path / "compiled", ["643E"], source_model=source)
    p = _pass({"package_name": "m", "create_shared_asset": False})

    with pytest.raises(ValueError, match="requires session_options"):
        p.run(model, str(tmp_path / "out"))


def test_duplicate_variant_names_fail_before_output(tmp_path, mock_model_package):
    model = _composite(tmp_path / "compiled", ["643E", "B03E"])
    p = _pass({"package_name": "m", "create_shared_asset": False, "variant_name_template": "{package_name}.{device}"})
    with pytest.raises(ValueError, match="duplicate variant names"):
        p.run(model, str(tmp_path / "out"))
    assert not (tmp_path / "out" / "m.ortpackage").exists()


def test_unsafe_variant_name_is_rejected(tmp_path, mock_model_package):
    model = _composite(tmp_path / "compiled", ["643E"])
    p = _pass({"package_name": "m", "create_shared_asset": False, "variant_name_template": "../{variant}"})
    with pytest.raises(ValueError, match="single safe path component"):
        p.run(model, str(tmp_path / "out"))


def test_refuses_non_empty_package_dir(tmp_path, mock_model_package):
    model = _composite(tmp_path / "compiled", ["643E"])
    stale = tmp_path / "out" / "m.ortpackage"
    stale.mkdir(parents=True)
    (stale / "stale.txt").write_text("old")

    with pytest.raises(FileExistsError, match="not empty"):
        _pass({"package_name": "m", "create_shared_asset": False}).run(model, str(tmp_path / "out"))


def test_commit_failure_does_not_publish_package(tmp_path, mock_model_package):
    model = _composite(tmp_path / "compiled", ["643E"])
    initial_commit = mock_model_package.commit.side_effect

    def commit(dest_root=None):
        if dest_root is None:
            raise RuntimeError("native commit failed")
        initial_commit(dest_root)

    mock_model_package.commit.side_effect = commit
    with (
        patch.object(ORTModelPackage, "_read_compat", return_value=""),
        pytest.raises(RuntimeError, match="native commit failed"),
    ):
        _pass({"package_name": "m", "create_shared_asset": False}).run(model, str(tmp_path / "out"))
    assert not (tmp_path / "out" / "m.ortpackage").exists()
    assert not list((tmp_path / "out").glob(".ortpackage-*"))


def test_check_ort_version_rejects_older_ort():
    ORTModelPackage._check_ort_version("1.32.0")
    ORTModelPackage._check_ort_version("1.32.1")
    with pytest.raises(RuntimeError, match="requires ONNX Runtime >= 1.32.0"):
        ORTModelPackage._check_ort_version("1.31.1")


def test_read_compat_uses_public_ort_api(tmp_path):
    ctx_model = tmp_path / "model_ctx.onnx"
    with patch("onnxruntime.get_compatibility_info_from_model", return_value="compat", create=True) as get_compat:
        assert ORTModelPackage._read_compat(ctx_model, _EP) == "compat"
    get_compat.assert_called_once_with(str(ctx_model), _EP)


# ─── _external_data_files ────────────────────────────────────────────────────


def test_external_data_files_preserves_relative_location(tmp_path):
    model = _make_model(tmp_path / "model.onnx", external_location="weights/model.data")
    files = ORTModelPackage._external_data_files(model)
    assert [loc for loc, _ in files] == ["weights/model.data"]
    assert files[0][1] == tmp_path / "weights" / "model.data"


def test_external_data_files_rejects_missing_file(tmp_path):
    model = _make_model(tmp_path / "model.onnx")
    (tmp_path / "weights.data").unlink()
    with pytest.raises(FileNotFoundError):
        ORTModelPackage._external_data_files(model)


@pytest.mark.parametrize("location", ["../victim.data", "..\\victim.data", "/tmp/victim.data", "C:\\victim.data"])
def test_external_locations_reject_unsafe_paths(location):
    with pytest.raises(ValueError, match="safe relative path"):
        ORTModelPackage._safe_relative_location(location)


def test_external_data_files_includes_nested_subgraph_initializers(tmp_path):
    path = tmp_path / "model.onnx"
    weights_path = tmp_path / "subgraph.data"
    weights_path.write_bytes(b"\0" * 4)
    weights = TensorProto()
    weights.name = "W"
    weights.data_type = TensorProto.FLOAT
    weights.dims.extend([1])
    weights.data_location = TensorProto.EXTERNAL
    for key, value in (("location", "subgraph.data"), ("offset", "0"), ("length", "4")):
        entry = weights.external_data.add()
        entry.key, entry.value = key, value

    then_branch = helper.make_graph(
        [], "then", [], [helper.make_tensor_value_info("W", TensorProto.FLOAT, [1])], initializer=[weights]
    )
    else_branch = helper.make_graph(
        [],
        "else",
        [],
        [helper.make_tensor_value_info("W", TensorProto.FLOAT, [1])],
        initializer=[helper.make_tensor("W", TensorProto.FLOAT, [1], [0.0])],
    )
    graph = helper.make_graph(
        [helper.make_node("If", ["cond"], ["out"], then_branch=then_branch, else_branch=else_branch)],
        "g",
        [helper.make_tensor_value_info("cond", TensorProto.BOOL, [])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx.save(model, str(path))

    assert ORTModelPackage._external_data_files(path) == [("subgraph.data", weights_path)]


# ─── EPContext sidecar files ─────────────────────────────────────────────────


def test_compiled_context_auxiliary_files_follow_ep_cache_context_reference(tmp_path):
    ctx_dir = tmp_path / "compiled"
    (ctx_dir / "blobs").mkdir(parents=True)
    (ctx_dir / "blobs" / "compiled.bin").write_bytes(b"context")
    node = helper.make_node(
        "EPContext", [], [], domain="com.microsoft", ep_cache_context="blobs/compiled.bin", embed_mode=0
    )
    graph = helper.make_graph([node], "ctx", [], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)])
    ctx_path = ctx_dir / "renamed_ctx.onnx"
    onnx.save(model, str(ctx_path))

    assert ORTModelPackage._compiled_context_auxiliary_files(ctx_path) == [
        ("blobs/compiled.bin", ctx_dir / "blobs" / "compiled.bin")
    ]
    output_ctx = tmp_path / "package" / "variant" / "model_ctx.onnx"
    ORTModelPackage._copy_variant(ctx_path, output_ctx)
    assert (output_ctx.parent / "blobs" / "compiled.bin").read_bytes() == b"context"
    assert output_ctx.is_file()


def test_compiled_context_auxiliary_files_reject_unsafe_ep_cache_context(tmp_path):
    node = helper.make_node(
        "EPContext", [], [], domain="com.microsoft", ep_cache_context="../outside.bin", embed_mode=0
    )
    graph = helper.make_graph([node], "ctx", [], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("com.microsoft", 1)])
    ctx_path = tmp_path / "model_ctx.onnx"
    onnx.save(model, str(ctx_path))

    with pytest.raises(ValueError, match="safe relative path"):
        ORTModelPackage._compiled_context_auxiliary_files(ctx_path)


# ─── _add_shared_asset ───────────────────────────────────────────────────────


def test_add_shared_asset_preserves_nested_layout(tmp_path, native_model_package):
    model = _make_model(tmp_path / "model.onnx", external_location="weights/model.data")
    package_dir = tmp_path / "pkg"
    with native_model_package() as package:
        uri = ORTModelPackage._add_shared_asset(model, package, tmp_path / "asset")
        package.commit(str(package_dir))

    assert uri.startswith("sha256:")
    asset_dir = package_dir / "shared_assets" / f"sha256-{uri.split(':')[1]}"
    # nested external-data location is preserved, not flattened to the basename
    assert (asset_dir / "model.onnx").is_file()
    assert (asset_dir / "weights" / "model.data").is_file()
    assert not (asset_dir / "model.data").exists()


def test_add_asset_entry_detects_collision():
    entries = {}
    ORTModelPackage._add_asset_entry(entries, "w.data", Path("a/w.data"))
    # same file + location is idempotent
    ORTModelPackage._add_asset_entry(entries, "w.data", Path("a/w.data"))
    # distinct file mapping to the same location is rejected
    with pytest.raises(ValueError, match="same package location"):
        ORTModelPackage._add_asset_entry(entries, "w.data", Path("b/w.data"))
