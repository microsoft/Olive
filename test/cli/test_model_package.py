# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import pytest

from olive.cli.model_package import ModelPackageCommand


def _create_source_dir(tmp_path, name, model_attributes, model_type="ONNXModel"):
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)

    model_config = {
        "type": model_type,
        "config": {
            "model_path": str(source_dir),
            "model_attributes": model_attributes,
        },
    }
    with open(source_dir / "model_config.json", "w") as f:
        json.dump(model_config, f)

    # Create a dummy model file
    (source_dir / "model_ctx.onnx").write_text("dummy")
    (source_dir / "model_ctx_QnnHtp_ctx.bin").write_text("dummy")

    return source_dir


class TestModelPackageCommand:
    def _run_command(self, args):
        from argparse import ArgumentParser

        parser = ArgumentParser()
        commands_parser = parser.add_subparsers()
        ModelPackageCommand.register_subcommand(commands_parser)
        parsed_args, unknown = parser.parse_known_args(args)
        cmd = parsed_args.func(parser, parsed_args, unknown)
        cmd.run()

    def test_merge_two_targets(self, tmp_path):
        """Test merging two context binary outputs."""
        # setup
        soc_60 = _create_source_dir(
            tmp_path,
            "soc_60",
            {
                "ep": "QNNExecutionProvider",
                "device": "NPU",
                "architecture": "60",
                "precision": "int4",
            },
        )
        soc_73 = _create_source_dir(
            tmp_path,
            "soc_73",
            {
                "ep": "QNNExecutionProvider",
                "device": "NPU",
                "architecture": "73",
                "precision": "int4",
            },
        )
        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(soc_60),
                "--source",
                str(soc_73),
                "-o",
                str(output_dir),
            ]
        )

        # assert: manifest.json has exactly the expected fields (no component_models for non-composite)
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        with open(manifest_path) as f:
            manifest = json.load(f)
        assert set(manifest.keys()) == {"name", "model_version", "task"}
        assert manifest["name"] == "output"
        assert manifest["model_version"] == "1.0"
        assert manifest["task"] == []

        # assert: model_variants in per-component metadata.json
        with open(output_dir / "models" / "output" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert "soc_60" in variants
        assert "soc_73" in variants
        assert variants["soc_60"]["file"] == str(soc_60)
        assert variants["soc_60"]["constraints"]["ep"] == "QNNExecutionProvider"
        assert variants["soc_60"]["constraints"]["device"] == "NPU"
        assert variants["soc_60"]["constraints"]["architecture"] == "60"
        assert variants["soc_73"]["constraints"]["architecture"] == "73"

        # assert: metadata.json exists in component directory
        metadata_path = output_dir / "models" / "output" / "metadata.json"
        assert metadata_path.exists()

        # assert: model files copied into component dir
        assert (output_dir / "models" / "output" / "soc_60" / "model_ctx.onnx").exists()
        assert (output_dir / "models" / "output" / "soc_73" / "model_ctx.onnx").exists()

    def test_merge_infer_name_from_dir(self, tmp_path):
        """Test that target name is inferred from directory name when not specified."""
        # setup
        soc_60 = _create_source_dir(
            tmp_path,
            "soc_60",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        soc_73 = _create_source_dir(
            tmp_path,
            "soc_73",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(soc_60),
                "--source",
                str(soc_73),
                "-o",
                str(output_dir),
            ]
        )

        # assert: variant file paths match the source directories
        with open(output_dir / "models" / "output" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["soc_60"]["file"] == str(soc_60)
        assert variants["soc_73"]["file"] == str(soc_73)

    def test_merge_openvino_targets(self, tmp_path):
        """Test merging OpenVINO context binary outputs."""
        # setup
        ov_2025_1 = _create_source_dir(
            tmp_path,
            "ov_2025.1",
            {
                "ep": "OpenVINOExecutionProvider",
                "device": "NPU",
                "sdk_version": "2025.1",
                "architecture": "NPU",
            },
        )
        ov_2025_2 = _create_source_dir(
            tmp_path,
            "ov_2025.2",
            {
                "ep": "OpenVINOExecutionProvider",
                "device": "NPU",
                "sdk_version": "2025.2",
                "architecture": "NPU",
            },
        )
        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(ov_2025_1),
                "--source",
                str(ov_2025_2),
                "-o",
                str(output_dir),
            ]
        )

        # assert: both OV targets present with correct constraints
        with open(output_dir / "models" / "output" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert len(variants) == 2
        assert variants["ov_2025.1"]["constraints"]["ep"] == "OpenVINOExecutionProvider"
        assert variants["ov_2025.1"]["constraints"]["device"] == "NPU"
        assert variants["ov_2025.2"]["constraints"]["device"] == "NPU"

    def test_merge_rejects_single_source(self, tmp_path):
        """Test that merging with a single source raises an error."""
        # setup
        soc_60 = _create_source_dir(
            tmp_path,
            "soc_60",
            {"ep": "QNNExecutionProvider"},
        )

        # execute + assert
        with pytest.raises(ValueError, match="At least two"):
            self._run_command(
                [
                    "generate-model-package",
                    "--source",
                    str(soc_60),
                    "-o",
                    str(tmp_path / "output"),
                ]
            )

    def test_merge_rejects_missing_model_config(self, tmp_path):
        """Test that merging rejects a directory without model_config.json."""
        # setup: one dir without model_config.json
        source_dir = tmp_path / "no_config"
        source_dir.mkdir()
        another = _create_source_dir(
            tmp_path,
            "valid",
            {"ep": "QNNExecutionProvider"},
        )

        # execute + assert
        with pytest.raises(ValueError, match="model_config.json"):
            self._run_command(
                [
                    "generate-model-package",
                    "--source",
                    str(source_dir),
                    "--source",
                    str(another),
                    "-o",
                    str(tmp_path / "output"),
                ]
            )

    def test_merge_rejects_nonexistent_path(self, tmp_path):
        """Test that merging rejects a nonexistent path."""
        # setup
        valid = _create_source_dir(
            tmp_path,
            "valid",
            {"ep": "QNNExecutionProvider"},
        )

        # execute + assert
        with pytest.raises(ValueError, match="does not exist"):
            self._run_command(
                [
                    "generate-model-package",
                    "--source",
                    "/nonexistent/path",
                    "--source",
                    str(valid),
                    "-o",
                    str(tmp_path / "output"),
                ]
            )

    def test_merge_optional_fields_omitted(self, tmp_path):
        """Test that optional fields are omitted from manifest when not in model_attributes."""
        # setup: targets with only ep and device (no architecture or ep_compatibility_info)
        soc_60 = _create_source_dir(
            tmp_path,
            "soc_60",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        soc_73 = _create_source_dir(
            tmp_path,
            "soc_73",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(soc_60),
                "--source",
                str(soc_73),
                "-o",
                str(output_dir),
            ]
        )

        # assert: optional constraint fields are absent
        with open(output_dir / "models" / "output" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        for v in variants.values():
            assert "architecture" not in v["constraints"]
            assert "ep_compatibility_info" not in v["constraints"]

    def test_merge_ep_compatibility_from_onnx_metadata(self, tmp_path):
        """ep_compatibility_info is extracted from ONNX model metadata when not in model_attributes."""
        # setup: real ONNX models with ep_compatibility_info in model metadata
        import onnx
        from onnx import TensorProto, helper

        for name, ep_compat_value in [
            ("soc_60", "QNNExecutionProvider;version=0.1.0;soc=60"),
            ("soc_73", "QNNExecutionProvider;version=0.1.0;soc=73"),
        ]:
            source_dir = tmp_path / name
            source_dir.mkdir(parents=True)

            x = helper.make_tensor_value_info("X", TensorProto.FLOAT, [1])
            y = helper.make_tensor_value_info("Y", TensorProto.FLOAT, [1])
            node = helper.make_node("Identity", ["X"], ["Y"])
            graph = helper.make_graph([node], "test", [x], [y])
            onnx_model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
            onnx.helper.set_model_props(
                onnx_model,
                {"ep_compatibility_info.QNNExecutionProvider": ep_compat_value},
            )
            onnx.save(onnx_model, str(source_dir / "model_ctx.onnx"))

            model_config = {
                "type": "ONNXModel",
                "config": {
                    "model_path": str(source_dir),
                    "model_attributes": {"ep": "QNNExecutionProvider", "device": "NPU"},
                },
            }
            with open(source_dir / "model_config.json", "w") as f:
                json.dump(model_config, f)

        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(tmp_path / "soc_60"),
                "--source",
                str(tmp_path / "soc_73"),
                "-o",
                str(output_dir),
            ]
        )

        # assert: ep_compatibility_info extracted from ONNX metadata
        with open(output_dir / "models" / "output" / "metadata.json") as f:
            metadata = json.load(f)
        variants = metadata["model_variants"]
        assert variants["soc_60"]["constraints"]["ep_compatibility_info"] == "QNNExecutionProvider;version=0.1.0;soc=60"
        assert variants["soc_73"]["constraints"]["ep_compatibility_info"] == "QNNExecutionProvider;version=0.1.0;soc=73"

    def test_merge_copies_config_files_to_configs_dir(self, tmp_path):
        """All additional_files (genai_config, tokenizer, etc.) are moved to configs/."""
        # setup: source dirs with config files in additional_files
        for name in ("soc_60", "soc_73"):
            source_dir = tmp_path / name
            source_dir.mkdir(parents=True)
            (source_dir / "model_ctx.onnx").write_text("dummy")
            (source_dir / "genai_config.json").write_text('{"model": {}}')
            (source_dir / "chat_template.jinja").write_text("template")
            (source_dir / "tokenizer.json").write_text("{}")

            model_config = {
                "type": "ONNXModel",
                "config": {
                    "model_path": str(source_dir),
                    "model_attributes": {
                        "ep": "QNNExecutionProvider",
                        "device": "NPU",
                        "additional_files": [
                            str(source_dir / "genai_config.json"),
                            str(source_dir / "chat_template.jinja"),
                            str(source_dir / "tokenizer.json"),
                        ],
                    },
                },
            }
            with open(source_dir / "model_config.json", "w") as f:
                json.dump(model_config, f)

        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(tmp_path / "soc_60"),
                "--source",
                str(tmp_path / "soc_73"),
                "-o",
                str(output_dir),
            ]
        )

        # assert: config files in configs/
        assert (output_dir / "configs" / "genai_config.json").exists()
        assert (output_dir / "configs" / "chat_template.jinja").exists()
        assert (output_dir / "configs" / "tokenizer.json").exists()

        # assert: config files NOT in variant directories
        assert not (output_dir / "models" / "output" / "soc_60" / "genai_config.json").exists()
        assert not (output_dir / "models" / "output" / "soc_73" / "genai_config.json").exists()

    def test_merge_composite_model(self, tmp_path):
        """Test merging composite model sources into per-component layout."""
        # setup: two SoC targets, each with two ONNX components
        for name, arch in [("soc_60", "60"), ("soc_73", "73")]:
            source_dir = tmp_path / name
            source_dir.mkdir(parents=True)

            (source_dir / "context_ctx.onnx").write_text("dummy_ctx")
            (source_dir / "embeddings.onnx").write_text("dummy_embed")

            model_config = {
                "type": "CompositeModel",
                "config": {
                    "model_path": str(source_dir),
                    "model_component_names": ["context_ctx", "embedding"],
                    "model_components": [
                        {
                            "type": "ONNXModel",
                            "config": {
                                "model_path": str(source_dir),
                                "onnx_file_name": "context_ctx.onnx",
                            },
                        },
                        {
                            "type": "ONNXModel",
                            "config": {
                                "model_path": str(source_dir),
                                "onnx_file_name": "embeddings.onnx",
                            },
                        },
                    ],
                    "model_attributes": {
                        "ep": "QNNExecutionProvider",
                        "device": "NPU",
                        "architecture": arch,
                    },
                },
            }
            with open(source_dir / "model_config.json", "w") as f:
                json.dump(model_config, f)

        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(tmp_path / "soc_60"),
                "--source",
                str(tmp_path / "soc_73"),
                "-o",
                str(output_dir),
            ]
        )

        # assert: manifest.json has component_models as a list for composite models
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert set(manifest.keys()) == {"name", "model_version", "task", "component_models"}
        assert manifest["name"] == "output"
        assert manifest["model_version"] == "1.0"
        assert isinstance(manifest["component_models"], list)
        assert "context_ctx" in manifest["component_models"]
        assert "embedding" in manifest["component_models"]

        # assert: per-component directories exist
        ctx_dir = output_dir / "models" / "context_ctx"
        embed_dir = output_dir / "models" / "embedding"
        assert ctx_dir.is_dir()
        assert embed_dir.is_dir()

        # assert: per-component metadata.json with correct constraints
        with open(ctx_dir / "metadata.json") as f:
            ctx_metadata = json.load(f)
        assert ctx_metadata["name"] == "context_ctx"
        assert "soc_60" in ctx_metadata["model_variants"]
        assert "soc_73" in ctx_metadata["model_variants"]
        assert ctx_metadata["model_variants"]["soc_60"]["constraints"]["architecture"] == "60"

        # assert: ONNX files in correct SoC subdirectories
        assert (ctx_dir / "soc_60" / "context_ctx.onnx").exists()
        assert (ctx_dir / "soc_73" / "context_ctx.onnx").exists()
        assert (embed_dir / "soc_60" / "embeddings.onnx").exists()
        assert (embed_dir / "soc_73" / "embeddings.onnx").exists()

    def test_merge_with_model_version(self, tmp_path):
        """Test that --model_version is included in manifest."""
        # setup
        soc_60 = _create_source_dir(
            tmp_path,
            "soc_60",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        soc_73 = _create_source_dir(
            tmp_path,
            "soc_73",
            {"ep": "QNNExecutionProvider", "device": "NPU"},
        )
        output_dir = tmp_path / "output"

        # execute
        self._run_command(
            [
                "generate-model-package",
                "--source",
                str(soc_60),
                "--source",
                str(soc_73),
                "--model_version",
                "3.0",
                "-o",
                str(output_dir),
            ]
        )

        # assert
        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)
        assert manifest["model_version"] == "3.0"
