# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
import json
from argparse import ArgumentParser
from unittest.mock import patch

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


class TestRunConfig:
    """Tests for _get_run_config workflow config construction."""

    def test_builds_model_package_workflow(self, tmp_path):
        """Config has ModelPackageModel input, ModelPackage pass, and correct accelerator."""
        # setup
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider", "device": "NPU"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider", "device": "NPU"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        # execute
        config = cmd._get_run_config(str(tmp_path / "tmp"))

        # assert: input model
        assert config["input_model"]["type"] == "ModelPackageModel"
        assert len(config["input_model"]["target_models"]) == 2
        assert config["input_model"]["target_names"] == ["soc_60", "soc_73"]

        # assert: pass config
        assert config["passes"]["pkg"]["type"] == "ModelPackage"
        assert config["passes"]["pkg"]["model_version"] == "1.0"

        # assert: accelerator from source model_attributes
        accel = config["systems"]["local_system"]["accelerators"][0]
        assert accel["device"] == "npu"
        assert accel["execution_providers"] == ["QNNExecutionProvider"]

        # assert: output dir
        assert config["output_dir"] == str(tmp_path / "out")

    def test_custom_model_name_and_version(self, tmp_path):
        """CLI args --model_name and --model_version are forwarded to pass config."""
        # setup
        src1 = _create_source_dir(tmp_path, "t1", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "t2", {"ep": "QNNExecutionProvider"})
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

        # execute
        config = cmd._get_run_config(str(tmp_path / "tmp"))

        # assert
        assert config["passes"]["pkg"]["model_name"] == "my_model"
        assert config["passes"]["pkg"]["model_version"] == "2.0"

    def test_defaults_accelerator_when_no_attributes(self):
        """Falls back to CPUExecutionProvider/cpu when model_attributes is empty."""
        # setup + execute
        ep, device = ModelPackageCommand._extract_accelerator_info([{"type": "ONNXModel", "config": {}}])

        # assert
        assert ep == "CPUExecutionProvider"
        assert device == "cpu"


class TestRunDelegation:
    """Test that run() delegates to _run_workflow()."""

    def test_run_calls_workflow(self, tmp_path):
        # setup
        src1 = _create_source_dir(tmp_path, "soc_60", {"ep": "QNNExecutionProvider"})
        src2 = _create_source_dir(tmp_path, "soc_73", {"ep": "QNNExecutionProvider"})
        cmd = _make_command(["generate-model-package", "-s", str(src1), "-s", str(src2), "-o", str(tmp_path / "out")])

        # execute + assert
        with patch.object(cmd, "_run_workflow", return_value=None) as mock_workflow:
            cmd.run()
            mock_workflow.assert_called_once()
