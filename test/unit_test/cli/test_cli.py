# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import subprocess
import sys
import unittest
from unittest.mock import MagicMock, patch

import pytest

from olive.cli.launcher import main as cli_main


@pytest.mark.parametrize("console_script", [True, False])
@pytest.mark.parametrize(
    "command",
    ["run", "configure-qualcomm-sdk", "manage-aml-compute", "convert-adapters", "tune-session-params", "auto-opt"],
)
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
@patch("huggingface_hub.repo_exists", return_value=True)
def test_finetune_command(_, mock_tempdir, mock_run, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    mock_tempdir.return_value = tmpdir.resolve()
    workflow_output_dir = tmpdir / "output_model"
    workflow_output_dir.mkdir(parents=True)
    dummy_output = workflow_output_dir / "model_config.json"
    with open(dummy_output, "w") as f:
        json.dump({"dummy": "output"}, f)

    # setup
    model_id = "dummy-model-id"
    command_args = [
        "finetune",
        "-m",
        model_id,
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
    assert config["input_model"]["model_path"] == model_id
    assert {el.name for el in output_dir.iterdir()} == {dummy_output.name}


def test_session_params_tuning_command(tmp_path):
    from test.unit_test.utils import ONNX_MODEL_PATH

    # some directories
    output_dir = tmp_path / "output_dir"

    # setup
    command_args = [
        "tune-session-params",
        "-m",
        str(ONNX_MODEL_PATH),
        "--output_path",
        str(output_dir),
        "--providers_list",
        "CPUExecutionProvider",
    ]

    # execute
    # run in subprocess to avoid affecting other tests
    out = subprocess.run(["olive", *command_args], check=True, capture_output=True)

    # assert
    assert f"Inference session parameters are saved to {output_dir}" in out.stdout.decode("utf-8")
    with open(output_dir / "cpu-cpu.json") as f:
        infer_settings = json.load(f)
        assert infer_settings["execution_provider"] == ["CPUExecutionProvider"]
        assert infer_settings.keys() >= {"provider_options", "session_options"}


@patch("olive.workflows.run")
@patch("olive.cli.capture_onnx.tempfile.TemporaryDirectory")
@patch("huggingface_hub.repo_exists", return_value=True)
@pytest.mark.parametrize("use_model_builder", [True, False])
def test_capture_onnx_command(_, mock_tempdir, mock_run, use_model_builder, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    mock_tempdir.return_value = tmpdir.resolve()
    workflow_output_dir = tmpdir / "output_model"
    workflow_output_dir.mkdir(parents=True)
    dummy_output = workflow_output_dir / "model_config.json"
    with open(dummy_output, "w") as f:
        json.dump({"config": {"inference_settings": {"dummy-key": "dummy-value"}}}, f)

    model_id = "dummy-model-id"
    command_args = [
        "capture-onnx-graph",
        "-m",
        model_id,
        "-o",
        str(output_dir),
    ]

    if use_model_builder:
        command_args.extend(["--use_model_builder", "--precision", "int4"])

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == model_id
    assert "m" in config["passes"] if use_model_builder else "c" in config["passes"]
    assert {el.name for el in output_dir.iterdir()} == {dummy_output.name}


@pytest.mark.parametrize("test_set", [(None, "successfully"), (MagicMock(name="blob1"), "failed")])
@patch("azure.storage.blob.ContainerClient")
@patch("olive.cli.shared_cache.get_credentials", return_value="dummy-credentials")
def test_shared_cache_command(_, mock_container_client, test_set):
    # setup
    command_args = [
        "shared-cache",
        "--delete",
        "--account",
        "account",
        "--container",
        "container",
        "--model_hash",
        "model_hash",
    ]
    mock_blob = MagicMock(name="blob1")
    mock_container_client.return_value.list_blobs.side_effect = [[mock_blob], [test_set[0]]]

    # execute
    with unittest.TestCase().assertLogs(logger="olive.cli.shared_cache", level="INFO") as log:
        cli_main(command_args)

    # assert
    assert (test_set[1] in message for message in log.output), "Expected log message not found."
    mock_container_client.assert_called_once_with(
        account_url="https://account.blob.core.windows.net", container_name="container", credential="dummy-credentials"
    )
    mock_container_client().delete_blob.assert_called_once()


@pytest.mark.parametrize("algorithm_name", ["awq", "gptq", "rtn"])
@patch("olive.workflows.run")
@patch("olive.cli.finetune.tempfile.TemporaryDirectory")
@patch("huggingface_hub.repo_exists")
def test_quantize_command(mock_repo_exists, mock_tempdir, mock_run, algorithm_name, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    mock_repo_exists.return_value = True
    mock_tempdir.return_value = tmpdir.resolve()
    mock_run.return_value = {}

    workflow_output_dir = tmpdir / "output_model" / algorithm_name
    workflow_output_dir.mkdir(parents=True)
    model_config_path = workflow_output_dir / "model_config.json"
    with model_config_path.open("w") as f:
        f.write("{}")

    # setup
    command_args = [
        "quantize",
        "-m",
        "dummy_model",
        "--algorithm",
        algorithm_name,
        "-o",
        str(output_dir),
    ]

    if algorithm_name == "gptq":
        command_args += ["-d", "dummy_dataset"]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == "dummy_model"
    assert {el.name for el in output_dir.iterdir()} == {algorithm_name}


# TODO(anyone): Add tests for ManageAMLComputeCommand
# Test for ConvertAdaptersCommand is added as part of test/unit_test/passes/onnx/test_extract_adapters.py
