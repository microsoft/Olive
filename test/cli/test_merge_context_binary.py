# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json

import pytest

from olive.cli.merge_context_binary import MergeContextBinaryCommand


def _create_source_dir(tmp_path, name, model_attributes, model_type="ONNXModel"):
    """Helper to create a fake context binary output directory with model_config.json."""
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


class TestMergeContextBinaryCommand:
    def _run_command(self, args):
        """Helper to parse args and run the command."""
        from argparse import ArgumentParser

        parser = ArgumentParser()
        commands_parser = parser.add_subparsers()
        MergeContextBinaryCommand.register_subcommand(commands_parser)
        parsed_args, unknown = parser.parse_known_args(args)
        cmd = parsed_args.func(parser, parsed_args, unknown)
        cmd.run()

    def test_merge_two_targets(self, tmp_path):
        """Test merging two context binary outputs."""
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
        self._run_command([
            "merge-context-binary",
            "--source", f"soc_60={soc_60}",
            "--source", f"soc_73={soc_73}",
            "-o", str(output_dir),
        ])

        # Check manifest.json
        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()

        with open(manifest_path) as f:
            manifest = json.load(f)

        assert len(manifest["models"]) == 2
        assert manifest["models"][0]["ep"] == "QNNExecutionProvider"
        assert manifest["models"][0]["device"] == "NPU"
        assert manifest["models"][0]["architecture"] == "60"
        assert manifest["models"][0]["precision"] == "int4"
        assert manifest["models"][0]["model_path"] == "soc_60/"
        assert manifest["models"][1]["architecture"] == "73"
        assert manifest["models"][1]["model_path"] == "soc_73/"

        # Check files were copied
        assert (output_dir / "soc_60" / "model_ctx.onnx").exists()
        assert (output_dir / "soc_73" / "model_ctx.onnx").exists()

    def test_merge_infer_name_from_dir(self, tmp_path):
        """Test that target name is inferred from directory name when not specified."""
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
        self._run_command([
            "merge-context-binary",
            "--source", str(soc_60),
            "--source", str(soc_73),
            "-o", str(output_dir),
        ])

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert manifest["models"][0]["model_path"] == "soc_60/"
        assert manifest["models"][1]["model_path"] == "soc_73/"

    def test_merge_openvino_targets(self, tmp_path):
        """Test merging OpenVINO context binary outputs."""
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
        self._run_command([
            "merge-context-binary",
            "--source", f"ov_2025.1={ov_2025_1}",
            "--source", f"ov_2025.2={ov_2025_2}",
            "-o", str(output_dir),
        ])

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        assert len(manifest["models"]) == 2
        assert manifest["models"][0]["ep"] == "OpenVINOExecutionProvider"
        assert manifest["models"][0]["sdk_version"] == "2025.1"
        assert manifest["models"][1]["sdk_version"] == "2025.2"

    def test_merge_rejects_single_source(self, tmp_path):
        """Test that merging with a single source raises an error."""
        soc_60 = _create_source_dir(
            tmp_path, "soc_60", {"ep": "QNNExecutionProvider"},
        )

        with pytest.raises(ValueError, match="At least two"):
            self._run_command([
                "merge-context-binary",
                "--source", str(soc_60),
                "-o", str(tmp_path / "output"),
            ])

    def test_merge_rejects_missing_model_config(self, tmp_path):
        """Test that merging rejects a directory without model_config.json."""
        source_dir = tmp_path / "no_config"
        source_dir.mkdir()

        another = _create_source_dir(
            tmp_path, "valid", {"ep": "QNNExecutionProvider"},
        )

        with pytest.raises(ValueError, match="model_config.json"):
            self._run_command([
                "merge-context-binary",
                "--source", str(source_dir),
                "--source", str(another),
                "-o", str(tmp_path / "output"),
            ])

    def test_merge_rejects_nonexistent_path(self, tmp_path):
        """Test that merging rejects a nonexistent path."""
        valid = _create_source_dir(
            tmp_path, "valid", {"ep": "QNNExecutionProvider"},
        )

        with pytest.raises(ValueError, match="does not exist"):
            self._run_command([
                "merge-context-binary",
                "--source", "/nonexistent/path",
                "--source", str(valid),
                "-o", str(tmp_path / "output"),
            ])

    def test_merge_optional_fields_omitted(self, tmp_path):
        """Test that optional fields are omitted from manifest when not in model_attributes."""
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
        self._run_command([
            "merge-context-binary",
            "--source", str(soc_60),
            "--source", str(soc_73),
            "-o", str(output_dir),
        ])

        with open(output_dir / "manifest.json") as f:
            manifest = json.load(f)

        # precision, sdk_version, architecture should not be present
        assert "precision" not in manifest["models"][0]
        assert "sdk_version" not in manifest["models"][0]
        assert "architecture" not in manifest["models"][0]
