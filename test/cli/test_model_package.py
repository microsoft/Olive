# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
from argparse import ArgumentParser
from unittest.mock import patch

import pytest

from olive.cli.model_package import ModelPackageCommand


def _create_source_dir(tmp_path, name, model_attributes, model_type="ONNXModel", extra_config=None):
    """Create a fake Olive output directory with model_config.json."""
    source_dir = tmp_path / name
    source_dir.mkdir(parents=True)

    config = {
        "model_path": str(source_dir / "model.onnx"),
        "model_attributes": model_attributes,
    }
    if extra_config:
        config.update(extra_config)

    model_config = {"type": model_type, "config": config}
    (source_dir / "model_config.json").write_text(json.dumps(model_config))
    (source_dir / "model.onnx").write_text("dummy")
    return source_dir


def _make_command(args_list):
    """Create a ModelPackageCommand from a list of CLI args."""
    parser = ArgumentParser()
    commands_parser = parser.add_subparsers()
    ModelPackageCommand.register_subcommand(commands_parser)
    parsed_args, unknown = parser.parse_known_args(args_list)
    return parsed_args.func(parser, parsed_args, unknown)


class TestModelPackageCommandParseSources:
    """Tests for source directory validation."""

    def test_rejects_single_source(self, tmp_path):
        src = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src), "-o", str(tmp_path / "out")])
        with pytest.raises(ValueError, match="At least two"):
            cmd._parse_sources()

    def test_rejects_missing_model_config(self, tmp_path):
        no_config = tmp_path / "no_config"
        no_config.mkdir()
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", str(no_config), "-s", str(valid), "-o", str(tmp_path / "out")]
        )
        with pytest.raises(ValueError, match="model_config.json"):
            cmd._parse_sources()

    def test_rejects_nonexistent_path(self, tmp_path):
        valid = _create_source_dir(tmp_path, "valid", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(
            ["generate-model-package", "-s", "/nonexistent/path", "-s", str(valid), "-o", str(tmp_path / "out")]
        )
        with pytest.raises(ValueError, match="does not exist"):
            cmd._parse_sources()

    def test_parses_valid_sources(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])
        sources = cmd._parse_sources()
        assert len(sources) == 2
        assert sources[0] == ("soc_60", src1)
        assert sources[1] == ("soc_73", src2)


class TestModelPackageCommandAcceleratorInfo:
    """Tests for accelerator info extraction from model configs."""

    def test_extracts_qnn_info(self):
        target_models = [
            {"type": "ONNXModel", "config": {"model_attributes": {"ep": "QNNExecutionProvider", "device": "NPU"}}}
        ]
        ep, device = ModelPackageCommand._extract_accelerator_info(target_models)
        assert ep == "QNNExecutionProvider"
        assert device == "npu"

    def test_extracts_openvino_info(self):
        target_models = [
            {
                "type": "ONNXModel",
                "config": {"model_attributes": {"ep": "OpenVINOExecutionProvider", "device": "NPU"}},
            }
        ]
        ep, device = ModelPackageCommand._extract_accelerator_info(target_models)
        assert ep == "OpenVINOExecutionProvider"
        assert device == "npu"

    def test_defaults_when_no_attributes(self):
        target_models = [{"type": "ONNXModel", "config": {}}]
        ep, device = ModelPackageCommand._extract_accelerator_info(target_models)
        assert ep == "CPUExecutionProvider"
        assert device == "cpu"

    def test_defaults_when_empty_list(self):
        ep, device = ModelPackageCommand._extract_accelerator_info([])
        assert ep == "CPUExecutionProvider"
        assert device == "cpu"


class TestModelPackageCommandRunConfig:
    """Tests for workflow config construction."""

    def test_config_has_model_package_pass(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        config = cmd._get_run_config(str(tmp_path / "tmp"))

        assert config["passes"]["pkg"]["type"] == "ModelPackage"
        assert config["passes"]["pkg"]["model_version"] == "1.0"

    def test_config_input_model_is_model_package(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        config = cmd._get_run_config(str(tmp_path / "tmp"))

        assert config["input_model"]["type"] == "ModelPackageModel"
        assert len(config["input_model"]["target_models"]) == 2
        assert config["input_model"]["target_names"] == ["soc_60", "soc_73"]

    def test_config_accelerator_from_source(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "ov_2025_1", {"ep": "OpenVINOExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "ov_2025_2", {"ep": "OpenVINOExecutionProvider", "device": "NPU"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        config = cmd._get_run_config(str(tmp_path / "tmp"))

        accel = config["systems"]["local_system"]["accelerators"][0]
        assert accel["device"] == "npu"
        assert accel["execution_providers"] == ["OpenVINOExecutionProvider"]

    def test_config_custom_model_name_and_version(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        cmd = _make_command(
            [
                "generate-model-package",
                "-s",
                str(src1),
                "-s",
                str(src2),
                "--model_name",
                "my_model",
                "--model_version",
                "2.0",
                "-o",
                str(tmp_path / "out"),
            ]
        )

        config = cmd._get_run_config(str(tmp_path / "tmp"))

        assert config["passes"]["pkg"]["model_name"] == "my_model"
        assert config["passes"]["pkg"]["model_version"] == "2.0"

    def test_config_output_dir_matches_arg(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider"})
        output = tmp_path / "my_output"
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(output)])

        config = cmd._get_run_config(str(tmp_path / "tmp"))

        assert config["output_dir"] == str(output)


class TestModelPackageCommandRun:
    """Test that run() delegates to _run_workflow()."""

    def test_run_calls_workflow(self, tmp_path):
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        with patch.object(cmd, "_run_workflow", return_value=None) as mock_workflow:
            cmd.run()
            mock_workflow.assert_called_once()
