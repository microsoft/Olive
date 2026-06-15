# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the CompositeToOnnxPackage pass."""

import json
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ONNXModelHandler
from olive.model.handler.composite import CompositeModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.composite_to_onnx_package import CompositeToOnnxPackage


def _write_tiny_onnx_with_external_data(onnx_path: Path, data_filename: str = "model.onnx.data") -> None:
    """Write a minimal valid ONNX model whose single initializer lives in an external data sidecar.

    The initializer is sized above onnx's default external-data size threshold
    (1024 bytes) so the .data sidecar actually gets written. The model itself
    stays tiny (one Identity node) so the test fixture remains cheap.
    """
    # 1024 floats = 4096 bytes, well above the default 1024-byte threshold for
    # promoting an initializer to external storage.
    data = np.arange(1024, dtype=np.float32)
    init_tensor = numpy_helper.from_array(data, name="weight")
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1024])
    node = helper.make_node("Identity", inputs=["weight"], outputs=["y"])
    graph = helper.make_graph([node], "g", inputs=[], outputs=[output], initializer=[init_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])

    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(
        model,
        str(onnx_path),
        save_as_external_data=True,
        all_tensors_to_one_file=True,
        location=data_filename,
    )


def _write_tiny_inline_onnx(onnx_path: Path) -> None:
    """Write a minimal self-contained (no external data) ONNX model."""
    init_tensor = numpy_helper.from_array(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32), name="weight")
    output = helper.make_tensor_value_info("y", TensorProto.FLOAT, [4])
    node = helper.make_node("Identity", inputs=["weight"], outputs=["y"])
    graph = helper.make_graph([node], "g", inputs=[], outputs=[output], initializer=[init_tensor])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save_model(model, str(onnx_path))


def _make_nested_genai_package(
    root: Path,
    components: dict[str, str],
    *,
    with_external_data: bool = True,
) -> Path:
    """Build a fake nested ORT-GenAI package at ``root``.

    ``components`` maps genai_config component keys (e.g. ``decoder``) to the
    relative ONNX filename under ``root`` (e.g. ``decoder/model.onnx``). Each
    component file is a real (tiny) ONNX model so the pass exercises real
    external-data rewriting rather than file rename only.
    """
    root.mkdir(parents=True, exist_ok=True)

    model_section: dict[str, dict[str, str]] = {}
    for key, rel_path in components.items():
        component_file = root / rel_path
        if with_external_data:
            _write_tiny_onnx_with_external_data(component_file)
        else:
            _write_tiny_inline_onnx(component_file)
        model_section[key] = {"filename": rel_path}

    # Shared root-level sidecars.
    (root / "tokenizer.json").write_text("{}", encoding="utf-8")
    (root / "chat_template.jinja").write_text("template", encoding="utf-8")

    (root / "genai_config.json").write_text(
        json.dumps({"model": {"type": "gemma4", **model_section}}, indent=2),
        encoding="utf-8",
    )
    return root


def _make_composite_handler(root: Path, components: dict[str, str]) -> CompositeModelHandler:
    component_handlers = [ONNXModelHandler(model_path=str(root / rel_path)) for rel_path in components.values()]
    return CompositeModelHandler(
        model_components=component_handlers,
        model_component_names=list(components.keys()),
        model_path=str(root),
    )


class TestCompositeToOnnxPackage:
    def test_flattens_nested_package_to_root_level_filenames(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {
                "decoder": "decoder/model.onnx",
                "vision": "vision_encoder/model.onnx",
                "audio": "audio_encoder/model.onnx",
                "embedding": "embedding/model.onnx",
            },
        )
        composite = _make_composite_handler(
            src_root,
            {
                "decoder": "decoder/model.onnx",
                "vision_encoder": "vision_encoder/model.onnx",
                "audio_encoder": "audio_encoder/model.onnx",
                "embedding": "embedding/model.onnx",
            },
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        out_dir = Path(out.model_path).parent
        assert (out_dir / "decoder.onnx").is_file()
        assert (out_dir / "vision_encoder.onnx").is_file()
        assert (out_dir / "audio_encoder.onnx").is_file()
        assert (out_dir / "embedding.onnx").is_file()
        assert (out_dir / "genai_config.json").is_file()
        assert (out_dir / "tokenizer.json").is_file()
        assert (out_dir / "chat_template.jinja").is_file()

    def test_rewrites_genai_config_filenames(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "vision": "vision_encoder/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root,
            {
                "decoder": "decoder/model.onnx",
                "vision_encoder": "vision_encoder/model.onnx",
            },
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        rewritten = json.loads((Path(out.model_path).parent / "genai_config.json").read_text(encoding="utf-8"))
        assert rewritten["model"]["decoder"]["filename"] == "decoder.onnx"
        assert rewritten["model"]["vision"]["filename"] == "vision_encoder.onnx"

    def test_returns_onnx_handler_with_entry_point_next_to_genai_config(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "vision": "vision_encoder/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root,
            {
                "decoder": "decoder/model.onnx",
                "vision_encoder": "vision_encoder/model.onnx",
            },
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        assert isinstance(out, ONNXModelHandler)
        # Evaluator-style auto-detection: parent of model_path must contain genai_config.json
        parent = Path(out.model_path).parent
        assert (parent / "genai_config.json").is_file()
        assert Path(out.model_path).name == "decoder.onnx"

    def test_rewrites_external_data_location_to_new_filename(self, tmp_path):
        """External-data references inside each component ONNX must point at the renamed sidecar.

        Regression test for a real-world failure: hardlinking a .onnx file +
        its .data sidecar to new names left the embedded "location" pointer
        inside the proto pointing at the old name (e.g. "model.onnx.data"),
        causing ONNX Runtime to fail at load with "External data path does not
        exist". The pass must rewrite each component model's external-data
        location to match its new flat filename, and produce a real .data file
        with the new name alongside it.
        """
        src_root = _make_nested_genai_package(tmp_path / "src", {"decoder": "decoder/model.onnx"})
        composite = _make_composite_handler(src_root, {"decoder": "decoder/model.onnx"})

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        out_dir = Path(out.model_path).parent
        # Both the flat ONNX file and the matching renamed sidecar must exist.
        assert (out_dir / "decoder.onnx").is_file()
        assert (out_dir / "decoder.onnx.data").is_file()

        # The embedded external-data location inside the rewritten ONNX file
        # must reference the new sidecar name, not the source layout's
        # "model.onnx.data". Load without materializing external data so the
        # initializer keeps its ``external_data`` pointer rather than getting
        # the bytes inlined as ``raw_data``.
        proto_only = onnx.load(str(out_dir / "decoder.onnx"), load_external_data=False)
        weight = next(t for t in proto_only.graph.initializer if t.name == "weight")
        location_entries = [entry.value for entry in weight.external_data if entry.key == "location"]
        assert location_entries == ["decoder.onnx.data"], (
            f"expected location='decoder.onnx.data', got external_data={list(weight.external_data)}"
        )

        # And the bytes should actually load through that new pointer (catches
        # the case where the .data file was written under the right name but
        # corrupted, or vice versa).
        materialized = onnx.load(str(out_dir / "decoder.onnx"), load_external_data=True)
        loaded_weight = next(t for t in materialized.graph.initializer if t.name == "weight")
        loaded_array = numpy_helper.to_array(loaded_weight)
        assert loaded_array.shape == (1024,)
        assert loaded_array[0] == 0.0
        assert loaded_array[-1] == 1023.0

    def test_handles_inline_onnx_without_external_data(self, tmp_path):
        """Self-contained ONNX models (no .data sidecar) should still flatten correctly."""
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx"},
            with_external_data=False,
        )
        composite = _make_composite_handler(src_root, {"decoder": "decoder/model.onnx"})

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        out_dir = Path(out.model_path).parent
        assert (out_dir / "decoder.onnx").is_file()
        # No external-data sidecar should be present since the source had none.
        assert not (out_dir / "decoder.onnx.data").exists()

    def test_uses_fallback_entry_point_when_requested_one_missing(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"vision": "vision_encoder/model.onnx", "embedding": "embedding/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root,
            {
                "vision_encoder": "vision_encoder/model.onnx",
                "embedding": "embedding/model.onnx",
            },
        )

        # The default entry_point_component is "decoder", which doesn't exist here.
        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        assert Path(out.model_path).name in {"vision_encoder.onnx", "embedding.onnx"}

    def test_honors_explicit_entry_point_component(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "embedding": "embedding/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root,
            {
                "decoder": "decoder/model.onnx",
                "embedding": "embedding/model.onnx",
            },
        )

        p = create_pass_from_dict(
            CompositeToOnnxPackage,
            {"entry_point_component": "embedding"},
            disable_search=True,
        )
        out = p.run(composite, str(tmp_path / "out"))

        assert Path(out.model_path).name == "embedding.onnx"

    def test_rejects_package_without_genai_config(self, tmp_path):
        src_root = tmp_path / "src"
        src_root.mkdir()
        _write_tiny_inline_onnx(src_root / "decoder" / "model.onnx")
        composite = _make_composite_handler(src_root, {"decoder": "decoder/model.onnx"})

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        with pytest.raises(ValueError, match=r"genai_config\.json"):
            p.run(composite, str(tmp_path / "out"))

    def test_handles_unique_collision_in_subdir_names(self, tmp_path):
        # Two components living in subdirs with the same internal filename shouldn't collide.
        src_root = tmp_path / "src"
        src_root.mkdir()
        _write_tiny_inline_onnx(src_root / "model_a" / "model.onnx")
        _write_tiny_inline_onnx(src_root / "model_b" / "model.onnx")
        (src_root / "genai_config.json").write_text(
            json.dumps(
                {
                    "model": {
                        "first": {"filename": "model_a/model.onnx"},
                        "second": {"filename": "model_b/model.onnx"},
                    }
                }
            ),
            encoding="utf-8",
        )

        composite = CompositeModelHandler(
            model_components=[
                ONNXModelHandler(model_path=str(src_root / "model_a" / "model.onnx")),
                ONNXModelHandler(model_path=str(src_root / "model_b" / "model.onnx")),
            ],
            model_component_names=["first", "second"],
            model_path=str(src_root),
        )

        p = create_pass_from_dict(
            CompositeToOnnxPackage,
            {"entry_point_component": "first"},
            disable_search=True,
        )
        out = p.run(composite, str(tmp_path / "out"))

        out_dir = Path(out.model_path).parent
        assert (out_dir / "model_a.onnx").is_file()
        assert (out_dir / "model_b.onnx").is_file()
