# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import subprocess
import sys
from unittest.mock import patch

import pytest

from olive.cli.launcher import main as cli_main


@pytest.mark.parametrize("console_script", [True, False])
@pytest.mark.parametrize("command", ["run", "configure-qualcomm-sdk", "manage-aml-compute", "export-adapters"])
def test_valid_command(console_script, command):
    # setup
    command_args = []
    if console_script:
        command_args.append("olive")
    else:
        command_args.extend([sys.executable, "-m", "olive"])
    if command:
        command_args.append(command)
    command_args.append("--help")

    # execute
    out = subprocess.run(command_args, check=True, capture_output=True)

    # assert
    if not console_script:
        # the help message only says `python` when running as a module
        command_args[0] = "python"
    assert f"usage: {' '.join(command_args[:-1])}" in out.stdout.decode("utf-8")


@pytest.mark.parametrize("console_script", [True, False])
def test_invalid_command(console_script):
    # setup
    command_args = []
    if console_script:
        command_args.append("olive")
    else:
        command_args.extend([sys.executable, "-m", "olive"])
    command_args.append("invalid-command")

    # execute and assert
    with pytest.raises(subprocess.CalledProcessError):
        subprocess.run(command_args, check=True, capture_output=True)


@pytest.mark.parametrize("deprecated_module", ["olive.workflows.run", "olive.platform_sdk.qualcomm.configure"])
def test_legacy_call(deprecated_module):
    # execute
    out = subprocess.run([sys.executable, "-m", deprecated_module, "--help"], check=True, capture_output=True)

    # assert
    assert (
        f"Running `python -m {deprecated_module}` is deprecated and might be removed in the future."
        in out.stderr.decode("utf-8")
    )


@pytest.mark.parametrize("packages", [True, False])
@pytest.mark.parametrize("setup", [True, False])
@pytest.mark.parametrize("tempdir", [None, "tempdir"])
@patch("olive.workflows.run")
def test_workflow_run_command(mock_run, tempdir, setup, packages):
    # setup
    command_args = ["run", "--run-config", "config.json"]
    if packages:
        command_args.append("--packages")
    if setup:
        command_args.append("--setup")
    if tempdir is not None:
        command_args.extend(["--tempdir", tempdir])

    # execute
    cli_main(command_args)

    # assert
    mock_run.assert_called_once_with(
        run_config="config.json", setup=setup, package_config=None, tempdir=tempdir, packages=packages
    )


@patch("olive.platform_sdk.qualcomm.configure.configure.configure")
def test_configure_qualcomm_sdk_command(mock_configure):
    # setup
    command_args = ["configure-qualcomm-sdk", "--py_version", "3.6", "--sdk", "snpe"]

    # execute
    cli_main(command_args)

    # assert
    mock_configure.assert_called_once_with("3.6", "snpe")


@patch("olive.workflows.run")
@patch("olive.cli.finetune.tempfile.TemporaryDirectory")
def test_finetune_command(mock_tempdir, mock_run, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    mock_tempdir.return_value = tmpdir.resolve()
    workflow_output_dir = tmpdir / "f-c-o-e-m" / "gpu-cuda_model"
    workflow_output_dir.mkdir(parents=True)
    dummy_output = workflow_output_dir / "dummy_output"
    with open(dummy_output, "w") as f:
        f.write("dummy_output")

    # setup
    command_args = [
        "finetune",
        "-m",
        "dummy_model",
        "-d",
        "dummy_dataset",
        "--text_field",
        "dummy_text_field",
        "-o",
        str(output_dir),
    ]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == "dummy_model"
    assert {el.name for el in output_dir.iterdir()} == {dummy_output.name}


# TODO(anyone): Add tests for ManageAMLComputeCommand
# Test for ExportAdaptersCommand is added as part of test/unit_test/passes/onnx/test_export_adapters.py
