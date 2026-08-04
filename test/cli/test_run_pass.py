# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser

import pytest


def test_run_pass_command_help(capsys):
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["run-pass", "--help"])

    assert exc_info.value.code == 0
    help_text = capsys.readouterr().out
    assert "usage:" in help_text
    assert "run-pass" in help_text
    assert "--pass-name" in help_text
    assert "Name of the pass to run on the input model" in help_text


def test_run_pass_command_argument_parsing():
    """Test the argument parsing without full CLI execution."""
    # Create a minimal parser to test our command registration
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    # Import and register our command without triggering heavy dependencies
    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test parsing with valid arguments
    args = parser.parse_args(["run-pass", "--pass-name", "OnnxConversion", "-m", "test_model", "-o", "/tmp/output"])

    assert args.pass_name == "OnnxConversion"
    assert args.model_name_or_path == "test_model"
    assert not hasattr(args, "list_passes") or not args.list_passes


def test_run_pass_command_list_passes():
    """Test the --list-passes argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test list-passes argument
    args = parser.parse_args(["run-pass", "--list-passes"])
    assert args.list_passes is True


def test_run_pass_command_pass_config():
    """Test the --pass-config argument."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test pass-config argument
    json_config = '{"convert_attribute": true}'
    args = parser.parse_args(
        [
            "run-pass",
            "--pass-name",
            "OnnxConversion",
            "-m",
            "test_model",
            "-o",
            "/tmp/output",
            "--pass-config",
            json_config,
        ]
    )

    assert args.pass_config == json_config


def test_run_pass_command_missing_pass_name():
    """Test that the run-pass command requires --pass-name argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test missing pass-name without list-passes should succeed in parsing but fail later
    args = parser.parse_args(["run-pass", "-m", "test_model", "-o", "/tmp/output"])
    assert args.pass_name is None
    assert args.model_name_or_path == "test_model"


def test_run_pass_command_missing_model():
    """Test that the run-pass command requires model argument when not listing passes."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test missing model should succeed in parsing but fail later when required
    args = parser.parse_args(["run-pass", "--pass-name", "OnnxConversion", "-o", "/tmp/output"])
    assert args.pass_name == "OnnxConversion"
    assert getattr(args, "model_name_or_path", None) is None


def test_run_pass_command_device_options():
    """Test the --device and accelerator argument parsing."""
    parser = ArgumentParser()
    sub_parsers = parser.add_subparsers()

    from olive.cli.run_pass import RunPassCommand

    RunPassCommand.register_subcommand(sub_parsers)

    # Test device argument parsing
    args = parser.parse_args(
        [
            "run-pass",
            "--pass-name",
            "OnnxConversion",
            "-m",
            "test_model",
            "-o",
            "/tmp/output",
            "--device",
            "gpu",
            "--provider",
            "CUDAExecutionProvider",
        ]
    )

    assert args.device == "gpu"
    assert args.provider == "CUDAExecutionProvider"

    # Test default device (cpu)
    args_default = parser.parse_args(
        ["run-pass", "--pass-name", "OnnxConversion", "-m", "test_model", "-o", "/tmp/output"]
    )

    assert args_default.device == "cpu"
    assert args_default.provider == "CPUExecutionProvider"
