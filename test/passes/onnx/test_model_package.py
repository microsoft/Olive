# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.handler.model_package import ModelPackageModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.model_package import ModelPackage


def _make_onnx_handler(tmp_path, name="model", model_attributes=None):
    model_dir = tmp_path / name
    model_dir.mkdir(parents=True, exist_ok=True)
    model_file = model_dir / f"{name}.onnx"
    model_file.write_text("dummy")
    return ONNXModelHandler(model_path=str(model_file), model_attributes=model_attributes)


def _make_onnx_handler_with_metadata(tmp_path, name="model", model_attributes=None, onnx_metadata=None):
    """Create an ONNXModelHandler backed by a real ONNX model with custom metadata."""
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


def _make_model_package(tmp_path, target_configs):
    targets = []
    names = []
    for name, attrs in target_configs:
        handler = _make_onnx_handler(tmp_path, name=name, model_attributes=attrs)
        targets.append(handler)
        names.append(name)
    return ModelPackageModelHandler(targets, names, model_path=tmp_path, model_attributes={})


# ===========================================================================
# ModelPackage tests
# ===========================================================================


class TestModelPackage:
    def _create_packager(self, ep="QNNExecutionProvider", device="NPU", config=None):
        accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)
        return create_pass_from_dict(
            ModelPackage,
            config or {},
            disable_search=True,
            accelerator_spec=accelerator_spec,
        )

    def test_packager_generates_manifest(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [
                ("soc_60", {"device": "NPU"}),
                ("soc_73", {"device": "NPU"}),
            ],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        # assert: result type
        assert isinstance(result, ModelPackageModelHandler)

        # assert: manifest.json always has component_models
        manifest_path = tmp_path / "output" / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert set(manifest.keys()) == {"name", "model_version", "task", "component_models"}
        assert manifest["name"] == "output"
        assert manifest["model_version"] == "1.0"
        assert manifest["task"] == ""
        assert manifest["component_models"] == ["model"]

        # assert: metadata.json under task-derived component dir ("model" when no _name_or_path)
        metadata_path = tmp_path / "output" / "models" / "model" / "metadata.json"
        assert metadata_path.exists()
        with open(metadata_path) as f:
            metadata = json.load(f)
        assert set(metadata.keys()) == {"name", "model_variants"}
        assert metadata["name"] == "model"

        # assert: model_variants contain per-target constraints
        variants = metadata["model_variants"]
        assert "soc_60" in variants
        assert "soc_73" in variants
        assert variants["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == ""

    def test_packager_ep_compatibility_from_onnx_metadata(self, tmp_path):
        """ep_compatibility_info is extracted from ONNX model custom metadata."""
        # setup: real ONNX models with ep_compatibility_info in ONNX metadata
        h1 = _make_onnx_handler_with_metadata(
            tmp_path,
            "soc_60",
            model_attributes={},
            onnx_metadata={
                "ep_compatibility_info.QNNExecutionProvider": "QNNExecutionProvider;version=0.1.0;soc=60",
            },
        )
        h2 = _make_onnx_handler_with_metadata(
            tmp_path,
            "soc_73",
            model_attributes={},
            onnx_metadata={
                "ep_compatibility_info.QNNExecutionProvider": "QNNExecutionProvider;version=0.1.0;soc=73",
            },
        )
        mt = ModelPackageModelHandler([h1, h2], ["soc_60", "soc_73"], model_path=tmp_path)
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: ep_compatibility_info is extracted from ONNX metadata
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == "QNNExecutionProvider;version=0.1.0;soc=60"
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == "QNNExecutionProvider;version=0.1.0;soc=73"

    def test_packager_ep_compat_empty_without_onnx_metadata(self, tmp_path):
        """ep_compatibility_info is empty string when ONNX metadata has no such entry."""
        # setup: real ONNX models without ep_compatibility_info metadata
        h1 = _make_onnx_handler_with_metadata(
            tmp_path,
            "soc_60",
            model_attributes={},
            onnx_metadata={},
        )
        h2 = _make_onnx_handler_with_metadata(
            tmp_path,
            "soc_73",
            model_attributes={},
            onnx_metadata={},
        )
        mt = ModelPackageModelHandler([h1, h2], ["soc_60", "soc_73"], model_path=tmp_path)
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: ep_compatibility_info is empty string
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == ""
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == ""

    def test_packager_custom_model_name(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {}), ("soc_73", {})],
        )
        p = self._create_packager(config={"model_name": "my_model"})

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: manifest uses the custom name
        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["name"] == "my_model"
        assert set(manifest.keys()) == {"name", "model_version", "task", "component_models"}

        # assert: metadata.json uses task-derived component name (not model_name)
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "model"

    def test_packager_copies_config_files_to_configs_dir(self, tmp_path):
        """All additional_files (genai_config, tokenizer, etc.) are moved to configs/."""
        # setup: composite model with additional_files (config files)
        from olive.model import CompositeModelHandler

        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        (comp_dir / "model.onnx").write_text("dummy")
        (comp_dir / "genai_config.json").write_text('{"model": {}}')
        (comp_dir / "chat_template.jinja").write_text("template")
        (comp_dir / "tokenizer.json").write_text("{}")

        additional_files = [
            str(comp_dir / "genai_config.json"),
            str(comp_dir / "chat_template.jinja"),
            str(comp_dir / "tokenizer.json"),
        ]

        sub = ONNXModelHandler(model_path=str(comp_dir / "model.onnx"))
        comp = CompositeModelHandler(
            model_components=[sub],
            model_component_names=["part1"],
            model_path=str(comp_dir),
            model_attributes={"architecture": "60", "additional_files": additional_files},
        )
        mt = ModelPackageModelHandler([comp], ["soc_60"], model_path=tmp_path)
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: config files are in configs/
        pkg_root = tmp_path / "output"
        assert (pkg_root / "configs" / "genai_config.json").exists()
        assert (pkg_root / "configs" / "chat_template.jinja").exists()
        assert (pkg_root / "configs" / "tokenizer.json").exists()

        # assert: config files are NOT in the variant directory
        assert not (pkg_root / "models" / "output" / "soc_60" / "genai_config.json").exists()
        assert not (pkg_root / "models" / "output" / "soc_60" / "chat_template.jinja").exists()
        assert not (pkg_root / "models" / "output" / "soc_60" / "tokenizer.json").exists()

    def test_packager_copies_config_dirs_to_configs(self, tmp_path):
        """Directories in additional_files (e.g., openvino_tokenizer) are copied to configs/."""
        # setup: OV model with a tokenizer directory in additional_files
        variant_dir = tmp_path / "ov_target"
        variant_dir.mkdir()
        (variant_dir / "model.onnx").write_text("dummy")
        (variant_dir / "genai_config.json").write_text('{"model": {}}')
        tok_dir = variant_dir / "openvino_tokenizer"
        tok_dir.mkdir()
        (tok_dir / "tokenizer.xml").write_text("<xml/>")
        (tok_dir / "tokenizer.bin").write_bytes(b"\x00")

        additional_files = [
            str(variant_dir / "genai_config.json"),
            str(tok_dir),
        ]
        h = ONNXModelHandler(
            model_path=str(variant_dir / "model.onnx"),
            model_attributes={"architecture": "npu", "additional_files": additional_files},
        )
        mt = ModelPackageModelHandler([h], ["ov_2025_1"], model_path=tmp_path)
        p = self._create_packager(ep="OpenVINOExecutionProvider")

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: files and directories are in configs/
        pkg_root = tmp_path / "output"
        assert (pkg_root / "configs" / "genai_config.json").exists()
        assert (pkg_root / "configs" / "openvino_tokenizer").is_dir()
        assert (pkg_root / "configs" / "openvino_tokenizer" / "tokenizer.xml").exists()

        # assert: removed from variant directory
        assert not (pkg_root / "models" / "output" / "ov_2025_1" / "genai_config.json").exists()
        assert not (pkg_root / "models" / "output" / "ov_2025_1" / "openvino_tokenizer").exists()

    def test_packager_copies_base_model(self, tmp_path):
        """Base model (pre-context-binary ONNX files) are copied to models/<name>/base/."""
        # setup: base model directory with ONNX files and config files
        base_dir = tmp_path / "base_models"
        base_dir.mkdir()
        (base_dir / "embeddings.onnx").write_text("embed")
        (base_dir / "context_0.onnx").write_text("ctx0")
        (base_dir / "iterator_0.onnx").write_text("iter0")
        (base_dir / "lm_head.onnx").write_text("lmhead")
        (base_dir / "transformer_0.onnx.data").write_bytes(b"\x00" * 64)
        (base_dir / "genai_config.json").write_text('{"model": {}}')
        (base_dir / "tokenizer.json").write_text("{}")

        mt = _make_model_package(
            tmp_path,
            [
                (
                    "soc_60",
                    {
                        "architecture": "60",
                        "additional_files": [
                            str(base_dir / "genai_config.json"),
                            str(base_dir / "tokenizer.json"),
                        ],
                    },
                ),
            ],
        )
        mt.model_attributes = {"base_model_path": str(base_dir)}
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: model files are copied to base/
        pkg_root = tmp_path / "output"
        base_out = pkg_root / "models" / "model" / "base"
        assert base_out.is_dir()
        assert (base_out / "embeddings.onnx").exists()
        assert (base_out / "context_0.onnx").exists()
        assert (base_out / "iterator_0.onnx").exists()
        assert (base_out / "lm_head.onnx").exists()
        assert (base_out / "transformer_0.onnx.data").exists()

        # assert: config files are NOT in base/ (they belong in configs/)
        assert not (base_out / "genai_config.json").exists()
        assert not (base_out / "tokenizer.json").exists()
        assert (pkg_root / "configs" / "genai_config.json").exists()
        assert (pkg_root / "configs" / "tokenizer.json").exists()

        # assert: base variant is in metadata model_variants
        with open(pkg_root / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        assert "base" in metadata["model_variants"]
        assert metadata["model_variants"]["base"]["file"] == "context_0.onnx"
        assert metadata["model_variants"]["base"]["constraints"] == {}

    def test_packager_no_base_model_when_path_missing(self, tmp_path):
        """No base/ dir is created when base_model_path is not set."""
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {"architecture": "60"})],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert
        assert not (tmp_path / "output" / "models" / "model" / "base").exists()

    def test_packager_rejects_non_model_package(self, tmp_path):
        # setup
        handler = _make_onnx_handler(tmp_path, "single")
        p = self._create_packager()

        # execute + assert
        output_path = str(tmp_path / "output.onnx")
        with pytest.raises(AssertionError, match="requires a ModelPackageModelHandler"):
            p.run(handler, output_path)

    def test_packager_copies_files(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {}), ("soc_73", {})],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: variant directories exist under models/<component_name>/
        assert (tmp_path / "output" / "models" / "model" / "soc_60").is_dir()
        assert (tmp_path / "output" / "models" / "model" / "soc_73").is_dir()

    def test_packager_default_model_name_from_dir(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )
        p = self._create_packager()

        # execute: output path is "my_package.onnx", so dir name becomes "my_package"
        output_path = str(tmp_path / "my_package.onnx")
        p.run(mt, output_path)

        # assert: manifest name defaults to directory name
        with open(tmp_path / "my_package" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["name"] == "my_package"

    def test_packager_device_only_when_present(self, tmp_path):
        # setup: t1 has device="GPU", t2 has no device
        mt = _make_model_package(
            tmp_path,
            [("t1", {"device": "GPU"}), ("t2", {})],
        )
        p = self._create_packager(device="NPU")

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: device constraint only present for t1
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["t1"]["constraints"]["device"] == "GPU"
        assert "device" not in variants["t2"]["constraints"]

    def test_packager_constraints_without_device(self, tmp_path):
        # setup: targets with no device attribute
        mt = _make_model_package(
            tmp_path,
            [("t1", {}), ("t2", {})],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: device not in constraints, ep_compatibility_info is empty string
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        for v in variants.values():
            assert "device" not in v["constraints"]
            assert v["constraints"]["ep_compatibility_info"] == ""

    def test_packager_manifest_path_in_result_attributes(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        # assert: result model_attributes contains manifest_path
        assert "manifest_path" in result.model_attributes
        assert Path(result.model_attributes["manifest_path"]).name == "manifest.json"

    def test_packager_copy_skips_existing_dest(self, tmp_path):
        # setup: pre-create destination with a marker file
        mt = _make_model_package(
            tmp_path,
            [("t1", {"architecture": "a"}), ("t2", {"architecture": "b"})],
        )
        p = self._create_packager(config={"model_name": "mdl"})
        output_path = str(tmp_path / "output.onnx")
        output_dir = tmp_path / "output"
        component_dir = output_dir / "models" / "mdl"
        component_dir.mkdir(parents=True)
        (component_dir / "t1").mkdir()
        (component_dir / "t1" / "marker.txt").write_text("pre-existing")

        # execute
        p.run(mt, output_path)

        # assert: pre-existing file is not overwritten
        assert (component_dir / "t1" / "marker.txt").read_text() == "pre-existing"

    def test_packager_with_composite_model_handler(self, tmp_path):
        # setup: two SoC targets, each with two ONNX components (context_ctx, embedding)
        from olive.model import CompositeModelHandler

        comp_dir_1 = tmp_path / "comp1"
        comp_dir_1.mkdir()
        (comp_dir_1 / "context_ctx.onnx").write_text("dummy_ctx")
        (comp_dir_1 / "embeddings.onnx").write_text("dummy_embed")

        comp_dir_2 = tmp_path / "comp2"
        comp_dir_2.mkdir()
        (comp_dir_2 / "context_ctx.onnx").write_text("dummy_ctx")
        (comp_dir_2 / "embeddings.onnx").write_text("dummy_embed")

        sub1_ctx = ONNXModelHandler(model_path=str(comp_dir_1 / "context_ctx.onnx"))
        sub1_embed = ONNXModelHandler(model_path=str(comp_dir_1 / "embeddings.onnx"))
        sub2_ctx = ONNXModelHandler(model_path=str(comp_dir_2 / "context_ctx.onnx"))
        sub2_embed = ONNXModelHandler(model_path=str(comp_dir_2 / "embeddings.onnx"))

        comp1 = CompositeModelHandler(
            model_components=[sub1_ctx, sub1_embed],
            model_component_names=["context_ctx", "embedding"],
            model_path=str(comp_dir_1),
            model_attributes={"architecture": "60", "device": "NPU"},
        )
        comp2 = CompositeModelHandler(
            model_components=[sub2_ctx, sub2_embed],
            model_component_names=["context_ctx", "embedding"],
            model_path=str(comp_dir_2),
            model_attributes={"architecture": "73", "device": "NPU"},
        )
        mt = ModelPackageModelHandler([comp1, comp2], ["soc_60", "soc_73"], model_path=tmp_path)
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        result = p.run(mt, output_path)

        # assert: result type
        pkg_root = tmp_path / "output"
        assert isinstance(result, ModelPackageModelHandler)

        # assert: manifest.json has component_models as a list for composite models
        with open(pkg_root / "manifest.json") as f:
            manifest = json.load(f)
        assert set(manifest.keys()) == {"name", "model_version", "task", "component_models"}
        assert manifest["name"] == "output"
        assert manifest["model_version"] == "1.0"
        assert isinstance(manifest["component_models"], list)
        assert "context_ctx" in manifest["component_models"]
        assert "embedding" in manifest["component_models"]

        # assert: per-component directories with SoC subdirectories
        ctx_dir = pkg_root / "models" / "context_ctx"
        embed_dir = pkg_root / "models" / "embedding"
        assert ctx_dir.is_dir()
        assert embed_dir.is_dir()

        # assert: per-component metadata.json with correct fields
        with open(ctx_dir / "metadata.json") as f:
            ctx_metadata = json.load(f)
        assert set(ctx_metadata.keys()) == {"name", "model_variants"}
        assert ctx_metadata["name"] == "context_ctx"
        assert "soc_60" in ctx_metadata["model_variants"]
        assert "soc_73" in ctx_metadata["model_variants"]
        assert ctx_metadata["model_variants"]["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"
        assert ctx_metadata["model_variants"]["soc_60"]["constraints"]["ep_compatibility_info"] == ""

        with open(embed_dir / "metadata.json") as f:
            embed_metadata = json.load(f)
        assert embed_metadata["name"] == "embedding"
        assert "soc_60" in embed_metadata["model_variants"]
        assert "soc_73" in embed_metadata["model_variants"]

        # assert: ONNX files in correct SoC subdirectories
        assert (ctx_dir / "soc_60" / "context_ctx.onnx").exists()
        assert (ctx_dir / "soc_73" / "context_ctx.onnx").exists()
        assert (embed_dir / "soc_60" / "embeddings.onnx").exists()
        assert (embed_dir / "soc_73" / "embeddings.onnx").exists()

    def test_composite_packager_with_model_version(self, tmp_path):
        # setup: composite model with custom model_version
        from olive.model import CompositeModelHandler

        comp_dir = tmp_path / "comp"
        comp_dir.mkdir()
        (comp_dir / "model.onnx").write_text("dummy")

        sub = ONNXModelHandler(model_path=str(comp_dir / "model.onnx"))
        comp = CompositeModelHandler(
            model_components=[sub],
            model_component_names=["part1"],
            model_path=str(comp_dir),
            model_attributes={"architecture": "60"},
        )
        mt = ModelPackageModelHandler([comp], ["soc_60"], model_path=tmp_path)
        p = self._create_packager(config={"model_version": "2.5"})

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert
        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["model_version"] == "2.5"

    def test_manifest_includes_task_from_hf_hub(self, tmp_path):
        """Task is extracted via HuggingFace Hub API and component name is derived from it."""
        # setup: model with _name_or_path in attributes
        mt = _make_model_package(
            tmp_path,
            [
                ("soc_60", {"_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}),
                ("soc_73", {"_name_or_path": "Qwen/Qwen2.5-1.5B-Instruct"}),
            ],
        )
        p = self._create_packager()

        # mock the HF API call
        mock_info = type("MockInfo", (), {"pipeline_tag": "text-generation"})()
        with patch("olive.passes.onnx.model_package.model_info", return_value=mock_info):
            output_path = str(tmp_path / "output.onnx")
            p.run(mt, output_path)

        # assert: task mapped to component name "decoder"
        output_dir = tmp_path / "output"
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["task"] == "text_generation"
        assert manifest["component_models"] == ["decoder"]

        # assert: metadata.json under decoder/ directory
        assert (output_dir / "models" / "decoder" / "metadata.json").exists()
        with open(output_dir / "models" / "decoder" / "metadata.json") as f:
            metadata = json.load(f)
        assert metadata["name"] == "decoder"

    def test_manifest_empty_task_without_name_or_path(self, tmp_path):
        """Task is empty when no _name_or_path exists in model attributes."""
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {}), ("soc_73", {})],
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert
        with open(tmp_path / "output" / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["task"] == ""

    def test_packager_onnx_model_uses_filename_in_file_field(self, tmp_path):
        # setup
        mt = _make_model_package(
            tmp_path,
            [("soc_60", {})],
        )
        h2 = _make_onnx_handler(tmp_path, name="soc_73", model_attributes={})
        mt = ModelPackageModelHandler(
            [next(t for _, t in mt.get_target_models()), h2],
            ["soc_60", "soc_73"],
            model_path=tmp_path,
        )
        p = self._create_packager()

        # execute
        output_path = str(tmp_path / "output.onnx")
        p.run(mt, output_path)

        # assert: file field uses "<target>/<filename>.onnx" format
        with open(tmp_path / "output" / "models" / "model" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["soc_60"]["file"] == "soc_60.onnx"
        assert variants["soc_73"]["file"] == "soc_73.onnx"


# ===========================================================================
# Pass.run() model package auto-dispatch tests
# ===========================================================================


class TestPassRunModelPackage:
    def test_pass_run_iterates_targets(self, tmp_path):
        """A pass that does NOT accept model package should iterate over each target independently."""
        # setup
        from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16

        h1 = _make_onnx_handler(tmp_path, "t1", model_attributes={"architecture": "60"})
        h2 = _make_onnx_handler(tmp_path, "t2", model_attributes={"architecture": "73"})
        mt = ModelPackageModelHandler([h1, h2], ["t1", "t2"], model_path=tmp_path)
        accelerator_spec = AcceleratorSpec(accelerator_type="NPU", execution_provider="QNNExecutionProvider")

        # execute: mock _run_for_config to avoid real ONNX ops
        with patch.object(OnnxFloatToFloat16, "_run_for_config") as mock_run:

            def side_effect(model, config, output_model_path):
                out_file = Path(output_model_path)
                out_file.parent.mkdir(parents=True, exist_ok=True)
                out_file.write_text("dummy")
                return ONNXModelHandler(model_path=str(out_file), model_attributes=model.model_attributes)

            mock_run.side_effect = side_effect

            p = create_pass_from_dict(OnnxFloatToFloat16, {}, disable_search=True, accelerator_spec=accelerator_spec)
            output_path = str(tmp_path / "output.onnx")
            result = p.run(mt, output_path)

        # assert: result is still ModelPackageModelHandler, _run_for_config called once per target
        assert isinstance(result, ModelPackageModelHandler)
        assert result.target_names == ["t1", "t2"]
        assert mock_run.call_count == 2
