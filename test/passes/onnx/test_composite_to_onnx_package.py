# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for the CompositeToOnnxPackage pass.

The pass repackages a nested multi-component CompositeModel ORT-GenAI package as a
single ONNXModelHandler pointing at the entry-point component, preserving the nested
directory layout (no flattening / external-data rewriting).
"""

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
    """Write a minimal valid ONNX model whose single initializer lives in an external data sidecar."""
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


def _make_nested_genai_package(root: Path, components: dict[str, str]) -> Path:
    """Build a fake nested ORT-GenAI package at ``root``.

    ``components`` maps genai_config component keys (e.g. ``decoder``) to the
    relative ONNX filename under ``root`` (e.g. ``decoder/model.onnx``).
    """
    root.mkdir(parents=True, exist_ok=True)

    model_section: dict[str, dict[str, str]] = {}
    for key, rel_path in components.items():
        _write_tiny_onnx_with_external_data(root / rel_path)
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


def _find_genai_config_upward(onnx_file: Path, levels: int = 3) -> Path | None:
    """Mirror the evaluator's upward search for genai_config.json from an ONNX file."""
    candidate = onnx_file.parent
    for _ in range(levels):
        if (candidate / "genai_config.json").is_file():
            return candidate / "genai_config.json"
        if candidate.parent == candidate:
            break
        candidate = candidate.parent
    return None


class TestCompositeToOnnxPackage:
    def test_preserves_nested_component_layout(self, tmp_path):
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

        pkg_root = Path(out.model_path).parents[1]
        # Nested layout preserved (components stay in their subdirectories).
        assert (pkg_root / "decoder" / "model.onnx").is_file()
        assert (pkg_root / "vision_encoder" / "model.onnx").is_file()
        assert (pkg_root / "audio_encoder" / "model.onnx").is_file()
        assert (pkg_root / "embedding" / "model.onnx").is_file()
        # Shared sidecars and config copied to the package root.
        assert (pkg_root / "genai_config.json").is_file()
        assert (pkg_root / "tokenizer.json").is_file()
        assert (pkg_root / "chat_template.jinja").is_file()

    def test_returns_onnx_handler_pointing_at_nested_entry(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "vision": "vision_encoder/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root, {"decoder": "decoder/model.onnx", "vision_encoder": "vision_encoder/model.onnx"}
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        assert isinstance(out, ONNXModelHandler)
        entry = Path(out.model_path)
        # Handler points at the nested decoder entry, which really exists.
        assert entry.is_file()
        assert entry.name == "model.onnx"
        assert entry.parent.name == "decoder"
        # genai_config.json is discoverable by searching upward from the entry ONNX file.
        assert _find_genai_config_upward(entry) is not None

    def test_does_not_rewrite_genai_config_filenames(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "vision": "vision_encoder/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root, {"decoder": "decoder/model.onnx", "vision_encoder": "vision_encoder/model.onnx"}
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        pkg_root = Path(out.model_path).parents[1]
        genai_config = json.loads((pkg_root / "genai_config.json").read_text(encoding="utf-8"))
        # Nested paths are preserved verbatim (no flattening rewrite).
        assert genai_config["model"]["decoder"]["filename"] == "decoder/model.onnx"
        assert genai_config["model"]["vision"]["filename"] == "vision_encoder/model.onnx"

    def test_honors_explicit_entry_point_component(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"decoder": "decoder/model.onnx", "embedding": "embedding/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root, {"decoder": "decoder/model.onnx", "embedding": "embedding/model.onnx"}
        )

        p = create_pass_from_dict(CompositeToOnnxPackage, {"entry_point_component": "embedding"}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        assert Path(out.model_path).parent.name == "embedding"

    def test_uses_fallback_entry_point_when_requested_one_missing(self, tmp_path):
        src_root = _make_nested_genai_package(
            tmp_path / "src",
            {"vision": "vision_encoder/model.onnx", "embedding": "embedding/model.onnx"},
        )
        composite = _make_composite_handler(
            src_root, {"vision_encoder": "vision_encoder/model.onnx", "embedding": "embedding/model.onnx"}
        )

        # 'decoder' (the default entry) is absent -> falls back to a present component.
        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        out = p.run(composite, str(tmp_path / "out"))

        assert Path(out.model_path).parent.name in {"vision_encoder", "embedding"}

    def test_raises_on_non_composite_input(self, tmp_path):
        _write_tiny_onnx_with_external_data(tmp_path / "model.onnx")
        onnx_model = ONNXModelHandler(model_path=str(tmp_path / "model.onnx"))

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        with pytest.raises(ValueError, match="expects a CompositeModelHandler"):
            p.run(onnx_model, str(tmp_path / "out"))

    def test_raises_when_entry_component_file_missing(self, tmp_path):
        src_root = _make_nested_genai_package(tmp_path / "src", {"decoder": "decoder/model.onnx"})
        composite = _make_composite_handler(src_root, {"decoder": "decoder/model.onnx"})
        # Remove the referenced component file so the config points at a missing file
        # (after building the handler, whose components assert file existence).
        (src_root / "decoder" / "model.onnx").unlink()

        p = create_pass_from_dict(CompositeToOnnxPackage, {}, disable_search=True)
        with pytest.raises(ValueError, match="not found"):
            p.run(composite, str(tmp_path / "out"))
