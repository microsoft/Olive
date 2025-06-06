# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from unittest.mock import patch

import pytest

from olive.cli.launcher import main as cli_main


def test_one_command_help():
    """Test that the one command shows help properly."""
    # setup
    command_args = [sys.executable, "-m", "olive", "one", "--help"]

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    help_text = out.stdout.decode("utf-8")
    assert "usage:" in help_text
    assert "one" in help_text
    assert "--pass-name" in help_text
    assert "Run a single pass on the input model" in help_text


def test_one_command_missing_pass_name():
    """Test that the one command requires --pass-name argument."""
    # setup
    command_args = ["one", "-m", "dummy_model", "-o", "/tmp/output"]

    # execute and assert
    with pytest.raises(SystemExit) as exc_info:
        cli_main(command_args)
    
    # Should exit with code 2 (argparse error)
    assert exc_info.value.code == 2


def test_one_command_missing_model():
    """Test that the one command requires model argument."""
    # setup  
    command_args = ["one", "--pass-name", "OnnxConversion", "-o", "/tmp/output"]

    # execute and assert
    with pytest.raises(SystemExit) as exc_info:
        cli_main(command_args)
    
    # Should exit with code 2 (argparse error)
    assert exc_info.value.code == 2


def test_one_command_invalid_pass_name():
    """Test that the one command validates pass names."""
    # setup
    command_args = ["one", "--pass-name", "InvalidPassName", "-m", "dummy_model", "-o", "/tmp/output"]

    # execute and assert
    with pytest.raises(ValueError) as exc_info:
        cli_main(command_args)
    
    # Should mention that the pass was not found
    assert "InvalidPassName" in str(exc_info.value)
    assert "not found" in str(exc_info.value)


@patch("olive.workflows.run")
def test_one_command_valid_execution(mock_run, tmp_path):
    """Test that the one command executes properly with valid arguments."""
    # setup
    output_path = tmp_path / "output"
    output_path.mkdir()
    
    command_args = [
        "one", 
        "--pass-name", "OnnxConversion",
        "-m", "hf-internal-testing/tiny-random-LlamaForCausalLM",
        "-o", str(output_path)
    ]

    # execute
    cli_main(command_args)

    # assert
    mock_run.assert_called_once()
    run_config = mock_run.call_args[0][0]
    
    # Check that the config has the expected structure
    assert "input_model" in run_config
    assert "passes" in run_config
    assert "onnxconversion" in run_config["passes"]
    assert run_config["passes"]["onnxconversion"]["type"] == "OnnxConversion"
    assert run_config["output_dir"] == str(output_path)