# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.model.handler.model_package import ModelPackageModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_package import ModelPackage


def _make_onnx_handler(tmp_path, name="model", model_attributes=None):
    """Create a dummy ONNXModelHandler with a text file as the .onnx file."""
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"
    model_file.write_text("dummy")
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


def _make_real_onnx_handler(tmp_path, name="model", model_attributes=None, onnx_metadata=None):
    """Create an ONNXModelHandler backed by a valid ONNX model with optional custom metadata."""
    import onnx
    from onnx import TensorProto, helper

    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"

    x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
    y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
    node = helper.make_node("Identity", ["X"], ["Y"])
    graph = helper.make_graph([node], "test", [x], [y])
    onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

    if onnx_metadata:
        onnx.helper.set_model_props(onnx_model, onnx_metadata)

    onnx.save(onnx_model, str(model_file))
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


def _make_model_package(tmp_path, target_configs, parent_attrs=None):
    """Build a ModelPackageModelHandler from (name, attrs) pairs."""
    targets = []
    names = []
    for name, attrs in target_configs:
        targets.append(_make_onnx_handler(tmp_path, name=name, model_attributes=attrs))
        names.append(name)
    return ModelPackageModelHandler(targets, names, model_path=tmp_path, model_attributes=parent_attrs or {})


def _make_composite_model_package(tmp_path, soc_configs, component_names):
    """Build a ModelPackageModelHandler wrapping CompositeModelHandlers.

    soc_configs: list of (soc_name, attrs) pairs.
    component_names: list of ONNX component names shared by each SoC target.
    """
    composites = []
    soc_names = []
    for soc_name, attrs in soc_configs:
        comp_dir = tmp_path / soc_name
        comp_dir.mkdir(parents=True, exist_ok=True)
        subs = []
        for comp_name in component_names:
            (comp_dir / f"{comp_name}.onnx").write_text("dummy")
            subs.append(ONNXModelHandler(model_path=str(comp_dir / f"{comp_name}.onnx")))
        composites.append(
            CompositeModelHandler(
                model_components=subs,
                model_component_names=component_names,
                model_path=str(comp_dir),
                model_attributes=attrs,
            )
        )
        soc_names.append(soc_name)
    return ModelPackageModelHandler(composites, soc_names, model_path=tmp_path)


def _create_pass(ep="QNNExecutionProvider", device="NPU", config=None):
    accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)
    return create_pass_from_dict(ModelPackage, config or {}, disable_search=True, accelerator_spec=accelerator_spec)


def _run_pass(model, tmp_path, ep="QNNExecutionProvider", device="NPU", config=None):
    """Run ModelPackage pass and return (result, output_dir)."""
    p = _create_pass(ep=ep, device=device, config=config)
    result = p.run(model, str(tmp_path / "output.onnx"))
    return result, tmp_path / "output"


def _read_manifest(output_dir):
    with open(output_dir / "manifest.json") as f:
        return json.load(f)


def _read_metadata(output_dir, component_name="model"):
    with open(output_dir / "models" / component_name / "metadata.json") as f:
        return json.load(f)


class TestSingleComponentPackaging:
    """Tests for packaging non-composite models (single ONNX per variant)."""

    def test_manifest_and_metadata_structure(self, tmp_path):
        """Manifest has all required fields; metadata has correct variants and constraints."""
        # setup
        mt = _make_model_package(tmp_path, [("soc_60", {"device": "NPU"}), ("soc_73", {"device": "NPU"})])

        # execute
        result, output_dir = _run_pass(mt, tmp_path)

        # assert
        assert isinstance(result, ModelPackageModelHandler)

        manifest = _read_manifest(output_dir)
        assert set(manifest.keys()) == {"name", "model_version", "task", "component_models"}
        assert manifest["model_version"] == "1.0"
        assert manifest["component_models"] == ["model"]

        metadata = _read_metadata(output_dir)
        assert metadata["name"] == "model"
        assert set(metadata["model_variants"].keys()) == {"soc_60", "soc_73"}
        assert metadata["model_variants"]["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"
        assert metadata["model_variants"]["soc_60"]["constraints"]["device"] == "NPU"
        assert metadata["model_variants"]["soc_60"]["constraints"]["ep_compatibility_info"] == ""

        assert (output_dir / "models" / "model" / "soc_60").is_dir()
        assert (output_dir / "models" / "model" / "soc_73").is_dir()

    def test_custom_model_name_and_version(self, tmp_path):
        """model_name and model_version pass configs override defaults."""
        # setup
        mt = _make_model_package(tmp_path, [("t1", {}), ("t2", {})])

        # execute
        _, output_dir = _run_pass(mt, tmp_path, config={"model_name": "my_model", "model_version": "2.5"})

        # assert
        manifest = _read_manifest(output_dir)
        assert manifest["name"] == "my_model"
        assert manifest["model_version"] == "2.5"
        assert _read_metadata(output_dir)["name"] == "model"

    def test_default_model_name_from_output_dir(self, tmp_path):
        """Model name defaults to the output directory name."""
        # setup
        mt = _make_model_package(tmp_path, [("t1", {}), ("t2", {})])
        p = _create_pass()

        # execute
        p.run(mt, str(tmp_path / "my_package.onnx"))

        # assert
        manifest = _read_manifest(tmp_path / "my_package")
        assert manifest["name"] == "my_package"

    def test_file_field_uses_onnx_filename_only(self, tmp_path):
        """Metadata file field contains only the ONNX filename, not the variant folder prefix."""
        # setup
        mt = _make_model_package(tmp_path, [("soc_60", {})])

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        metadata = _read_metadata(output_dir)
        assert metadata["model_variants"]["soc_60"]["file"] == "soc_60.onnx"

    def test_device_omitted_when_absent(self, tmp_path):
        """Device constraint is not included when model_attributes has no device."""
        # setup
        mt = _make_model_package(tmp_path, [("t1", {"device": "GPU"}), ("t2", {})])

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        variants = _read_metadata(output_dir)["model_variants"]
        assert variants["t1"]["constraints"]["device"] == "GPU"
        assert "device" not in variants["t2"]["constraints"]

    def test_rejects_non_model_package_input(self, tmp_path):
        """Pass rejects input that is not a ModelPackageModelHandler."""
        # setup
        handler = _make_onnx_handler(tmp_path, "single")
        p = _create_pass()

        # execute + assert
        with pytest.raises(AssertionError, match="requires a ModelPackageModelHandler"):
            p.run(handler, str(tmp_path / "output.onnx"))

    def test_copy_skips_existing_destination(self, tmp_path):
        """Pre-existing variant directory is not overwritten."""
        # setup
        mt = _make_model_package(tmp_path, [("t1", {}), ("t2", {})])
        dest = tmp_path / "output" / "models" / "model" / "t1"
        dest.mkdir(parents=True)
        (dest / "marker.txt").write_text("pre-existing")

        # execute
        _run_pass(mt, tmp_path)

        # assert
        assert (dest / "marker.txt").read_text() == "pre-existing"

    def test_result_attributes_has_manifest_path(self, tmp_path):
        """Result model_attributes includes manifest_path and clears temporary keys."""
        # setup
        mt = _make_model_package(tmp_path, [("t1", {}), ("t2", {})])

        # execute
        result, _ = _run_pass(mt, tmp_path)

        # assert
        assert Path(result.model_attributes["manifest_path"]).name == "manifest.json"
        assert "additional_files" not in result.model_attributes
        assert "base_model_path" not in result.model_attributes


class TestEpCompatibility:
    """Tests for ep_compatibility_info extraction from ONNX metadata."""

    def test_extracted_from_onnx_metadata(self, tmp_path):
        """ep_compatibility_info is read from ONNX model metadata_props."""
        # setup
        h1 = _make_real_onnx_handler(
            tmp_path,
            "soc_60",
            model_attributes={},
            onnx_metadata={"ep_compatibility_info.QNNExecutionProvider": "soc=60"},
        )
        h2 = _make_real_onnx_handler(
            tmp_path,
            "soc_73",
            model_attributes={},
            onnx_metadata={"ep_compatibility_info.QNNExecutionProvider": "soc=73"},
        )
        mt = ModelPackageModelHandler([h1, h2], ["soc_60", "soc_73"], model_path=tmp_path)

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        variants = _read_metadata(output_dir)["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == "soc=60"
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == "soc=73"

    def test_empty_string_when_no_onnx_metadata(self, tmp_path):
        """ep_compatibility_info defaults to empty string when ONNX has no such entry."""
        # setup
        h1 = _make_real_onnx_handler(tmp_path, "soc_60", onnx_metadata={})
        h2 = _make_real_onnx_handler(tmp_path, "soc_73", onnx_metadata={})
        mt = ModelPackageModelHandler([h1, h2], ["soc_60", "soc_73"], model_path=tmp_path)

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        variants = _read_metadata(output_dir)["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == ""
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == ""


class TestConfigFiles:
    """Tests for config file (additional_files) copying to configs/ directory."""

    def test_files_copied_to_configs_dir(self, tmp_path):
        """Regular files in additional_files are copied to configs/ and removed from variants."""
        # setup
        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        (comp_dir / "model.onnx").write_text("dummy")
        (comp_dir / "genai_config.json").write_text('{"model": {}}')
        (comp_dir / "tokenizer.json").write_text("{}")

        additional_files = [str(comp_dir / "genai_config.json"), str(comp_dir / "tokenizer.json")]
        h = ONNXModelHandler(
            model_path=str(comp_dir / "model.onnx"),
            model_attributes={"additional_files": additional_files},
        )
        mt = ModelPackageModelHandler([h], ["soc_60"], model_path=tmp_path)

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        assert (output_dir / "configs" / "genai_config.json").exists()
        assert (output_dir / "configs" / "tokenizer.json").exists()
        assert not (output_dir / "models" / "model" / "soc_60" / "genai_config.json").exists()

    def test_directories_copied_to_configs(self, tmp_path):
        """Directories in additional_files (e.g., openvino_tokenizer) are copied to configs/."""
        # setup
        variant_dir = tmp_path / "ov_target"
        variant_dir.mkdir()
        (variant_dir / "model.onnx").write_text("dummy")
        tok_dir = variant_dir / "openvino_tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.xml").write_text("<xml/>")

        h = ONNXModelHandler(
            model_path=str(variant_dir / "model.onnx"),
            model_attributes={"additional_files": [str(tok_dir)]},
        )
        mt = ModelPackageModelHandler([h], ["ov_2025_1"], model_path=tmp_path)

        # execute
        _, output_dir = _run_pass(mt, tmp_path, ep="OpenVINOExecutionProvider")

        # assert
        assert (output_dir / "configs" / "openvino_tokenizer" / "tokenizer.xml").exists()
        assert not (output_dir / "models" / "model" / "ov_2025_1" / "openvino_tokenizer").exists()


class TestBaseModel:
    """Tests for base (pre-optimized) model copying."""

    def test_base_model_copied_and_in_metadata(self, tmp_path):
        """Base model files are copied to base/ and listed in metadata model_variants."""
        # setup
        base_dir = tmp_path / "base_models"
        base_dir.mkdir()
        (base_dir / "embeddings.onnx").write_text("embed")
        (base_dir / "context_0.onnx").write_text("ctx0")
        (base_dir / "weights.onnx.data").write_bytes(b"\x00" * 64)
        (base_dir / "genai_config.json").write_text('{"model": {}}')

        mt = _make_model_package(
            tmp_path,
            [("soc_60", {"additional_files": [str(base_dir / "genai_config.json")]})],
            parent_attrs={"base_model_path": str(base_dir)},
        )

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert: model files copied, config files excluded
        base_out = output_dir / "models" / "model" / "base"
        assert (base_out / "embeddings.onnx").exists()
        assert (base_out / "context_0.onnx").exists()
        assert (base_out / "weights.onnx.data").exists()
        assert not (base_out / "genai_config.json").exists()

        # assert: base variant in metadata with empty constraints
        variants = _read_metadata(output_dir)["model_variants"]
        assert variants["base"]["file"] == "context_0.onnx"
        assert variants["base"]["constraints"] == {}

    def test_no_base_dir_when_path_missing(self, tmp_path):
        """No base/ directory is created when base_model_path is not set."""
        # setup
        mt = _make_model_package(tmp_path, [("soc_60", {})])

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        assert not (output_dir / "models" / "model" / "base").exists()


class TestCompositePackaging:
    """Tests for packaging composite models (multiple ONNX components per variant)."""

    def test_composite_manifest_and_per_component_metadata(self, tmp_path):
        """Composite model produces per-component dirs, metadata, and manifest with component_models."""
        # setup
        mt = _make_composite_model_package(
            tmp_path,
            soc_configs=[("soc_60", {"device": "NPU"}), ("soc_73", {"device": "NPU"})],
            component_names=["context_ctx", "embedding"],
        )

        # execute
        result, output_dir = _run_pass(mt, tmp_path)

        # assert: manifest
        assert isinstance(result, ModelPackageModelHandler)
        manifest = _read_manifest(output_dir)
        assert set(manifest["component_models"]) == {"context_ctx", "embedding"}
        assert manifest["model_version"] == "1.0"

        # assert: per-component metadata
        ctx_meta = _read_metadata(output_dir, "context_ctx")
        assert ctx_meta["name"] == "context_ctx"
        assert set(ctx_meta["model_variants"].keys()) == {"soc_60", "soc_73"}
        assert ctx_meta["model_variants"]["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"

        embed_meta = _read_metadata(output_dir, "embedding")
        assert embed_meta["name"] == "embedding"

        # assert: ONNX files in correct variant dirs
        assert (output_dir / "models" / "context_ctx" / "soc_60" / "context_ctx.onnx").exists()
        assert (output_dir / "models" / "embedding" / "soc_73" / "embedding.onnx").exists()

    def test_composite_custom_model_version(self, tmp_path):
        """model_version config works for composite models."""
        # setup
        mt = _make_composite_model_package(
            tmp_path,
            soc_configs=[("soc_60", {})],
            component_names=["part1"],
        )

        # execute
        _, output_dir = _run_pass(mt, tmp_path, config={"model_version": "3.0"})

        # assert
        assert _read_manifest(output_dir)["model_version"] == "3.0"


class TestTaskExtraction:
    """Tests for task extraction via HuggingFace Hub API."""

    def test_task_from_hf_hub_maps_to_component_name(self, tmp_path):
        """HF pipeline_tag is used for task and maps to component directory name."""
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {"_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"})],
        )

        # execute
        mock_info = type("MockInfo", (), {"pipeline_tag": "text-generation"})()
        with patch("olive.passes.onnx.model_package.model_info", return_value=mock_info):
            _, output_dir = _run_pass(mt, tmp_path)

        # assert
        manifest = _read_manifest(output_dir)
        assert manifest["task"] == "text_generation"
        assert manifest["component_models"] == ["decoder"]
        assert _read_metadata(output_dir, "decoder")["name"] == "decoder"

    def test_empty_task_without_name_or_path(self, tmp_path):
        """Task is empty string when _name_or_path is not in model attributes."""
        # setup
        mt = _make_model_package(tmp_path, [("soc_60", {})])

        # execute
        _, output_dir = _run_pass(mt, tmp_path)

        # assert
        manifest = _read_manifest(output_dir)
        assert manifest["task"] == ""
        assert manifest["component_models"] == ["model"]


class TestPassAutoDispatch:
    """Tests for Pass.run() auto-dispatch on ModelPackageModelHandler."""

    def test_non_accepting_pass_iterates_targets(self, tmp_path):
        """A pass without _accepts_model_package_model runs independently on each variant."""
        # setup
        from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16

        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = ModelPackageModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)
        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        # execute
        with patch.object(OnnxFloatToFloat16, "_run_for_config") as mock_run:

            def side_effect(model, config, output_model_path):
                out_file = Path(output_model_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("dummy")
                return ONNXModelHandler(model_path=str(out_file), model_attributes=model.model_attributes)

            mock_run.side_effect = side_effect
            p = create_pass_from_dict(OnnxFloatToFloat16, {}, disable_search=True, accelerator_spec=accelerator_spec)
            result = p.run(mt, str(tmp_path / "output.onnx"))

        # assert
        assert isinstance(result, ModelPackageModelHandler)
        assert result.target_names == ["t1", "t2"]
        assert mock_run.call_count == 2
