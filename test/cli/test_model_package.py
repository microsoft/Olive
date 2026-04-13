# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
from argparse import ArgumentParser

import pytest

from olive.cli.model_package import ModelPackageCommand


def _create_source_dir(tmp_path, name, model_attributes):
    """Create a fake Olive output directory with model_config.json and a dummy .onnx file."""
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)
    model_config = {
        "type": "ONNXModel",
        "config": {"model_path": str(source_dir / "model.onnx"), "model_attributes": model_attributes},
    }
    (source_dir / "model_config.json").write_text(json.dumps(model_config))
    (source_dir / "model.onnx").write_text("dummy")
    return source_dir


def _make_command(args_list):
    """Create a ModelPackageCommand instance from CLI args."""
    parser = ArgumentParser()
    commands_parser = parser.add_subparsers()
    ModelPackageCommand.register_subcommand(commands_parser)
    parsed_args, unknown = parser.parse_known_args(args_list)
    return parsed_args.func(parser, parsed_args, unknown)


class TestSourceValidation:
    """Tests for _parse_sources validation logic."""

    def test_rejects_single_source(self, tmp_path):
        # setup
        src = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])

        # execute + assert
        with pytest.raises(ValueError, match="At least two"):
            cmd._parse_sources()

    def test_rejects_missing_model_config(self, tmp_path):
        # setup
        no_config = tmp_path / "no_config"
        no_config.mkdir()
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", str(no_config), "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        # execute + assert
        with pytest.raises(ValueError, match="model_config.json"):
            cmd._parse_sources()

    def test_rejects_nonexistent_path(self, tmp_path):
        # setup
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", "/nonexistent/path", "-s", str(valid), "-o", str(tmp_path / "out")]
        )

        # execute + assert
        with pytest.raises(ValueError, match="does not exist"):
            cmd._parse_sources()

    def test_parses_two_valid_sources(self, tmp_path):
        # setup
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        # execute
        sources = cmd._parse_sources()

        # assert
        assert len(sources) == 2
        assert sources[0] == ("soc_60", src1)
        assert sources[1] == ("soc_73", src2)


class TestGeneratePackageSingle:
    """Tests for single-component model package generation."""

    def test_generates_manifest_and_metadata(self, tmp_path):
        """Package output should have manifest.json and metadata.json."""
        # setup
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        out_dir = tmp_path / "out"
        cmd = _make_command(
            [
                "generate-model-package",
                "-s",
                str(src1),
                "-s",
                str(src2),
                "-o",
                str(out_dir),
                "--model_name",
                "test_model",
                "--model_version",
                "2.0",
            ]
        )

        # execute
        cmd.run()

        # assert: manifest
        manifest_path = out_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text())
        assert manifest["name"] == "test_model"
        assert manifest["model_version"] == "2.0"
        assert "component_models" in manifest

        # assert: metadata in component dir
        component_name = manifest["component_models"][0]
        metadata_path = out_dir / "models" / component_name / "metadata.json"
        assert metadata_path.exists()
        metadata = json.loads(metadata_path.read_text())
        assert "soc_60" in metadata["model_variants"]
        assert "soc_73" in metadata["model_variants"]

        # assert: constraints
        for variant in metadata["model_variants"].values():
            assert variant["constraints"]["ep"] == "QNNExecutionProvider"
            assert variant["constraints"]["device"] == "NPU"


class TestAcceleratorInfo:
    """Test accelerator info extraction."""

    def test_defaults_accelerator_when_no_attributes(self):
        """Falls back to CPUExecutionProvider/cpu when model_attributes is empty."""
        # setup + execute
        ep, device = ModelPackageCommand._extract_accelerator_info([{"type": "ONNXModel", "config": {}}])

        # assert
        assert ep == "CPUExecutionProvider"
        assert device == "cpu"
