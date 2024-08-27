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
    "command", ["run", "configure-qualcomm-sdk", "manage-aml-compute", "export-adapters", "tune-session-params"]
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
def test_finetune_command(mock_tempdir, mock_run, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    mock_tempdir.return_value = tmpdir.resolve()
    workflow_output_dir = tmpdir / "f-c-o-e" / "gpu-cuda_model"
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


@patch("olive.cli.perf_tuning.olive_run")
@patch("olive.cli.perf_tuning.tempfile.TemporaryDirectory")
@patch("olive.common.ort_inference.get_ort_inference_session")
@pytest.mark.parametrize("data_config_path", ["", "dummy_data_config_path.json"])
def test_perf_tuning_command(mock_ort_infer_sess, mock_tempdir, mock_run, data_config_path, tmp_path):
    # some directories
    tmpdir = tmp_path / "tmpdir"
    tmpdir.mkdir()

    output_dir = tmp_path / "output_dir"

    # setup
    data_config = {
        "name": "dummy_data",
        "type": "TransformersTokenDummyDataContainer",
        "load_dataset_config": {"model_name": "microsoft/phi-2"},
    }
    if data_config_path:
        data_config_path = str(tmpdir / data_config_path)
        with open(data_config_path, "w") as f:
            json.dump(data_config, f)
    mock_tempdir.return_value = tmpdir.resolve()
    workflow_output_dir = tmpdir / "perf_tuning" / "gpu-cuda-model"
    workflow_output_dir.mkdir(parents=True)
    dummy_output = workflow_output_dir / "dummy_output"
    with open(dummy_output, "w") as f:
        f.write("dummy_output")

    # setup
    command_args = [
        "tune-session-params",
        "--model",
        "dummy_model",
        "--data_config_path",
        data_config_path,
        "--hf_model_name",
        "Intel/bert-base-uncased-mrpc",
        "--output_path",
        str(output_dir),
    ]

    # execute
    cli_main(command_args)

    config = mock_run.call_args[0][0]
    assert config["input_model"]["model_path"] == "dummy_model"
    assert config["data_configs"][0].name == config["passes"]["perf_tuning"]["data_config"]


@patch("olive.workflows.run")
@patch("olive.cli.capture_onnx.tempfile.TemporaryDirectory")
@pytest.mark.parametrize("use_model_builder", [True, False])
def test_capture_onnx_command(mock_tempdir, mock_run, use_model_builder, tmp_path):
    # setup
    mock_tempdir.return_value = tmp_path.resolve()
    output_dir = tmp_path / "output_dir"
    model_id = "microsoft/phi-2"

    # setup
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


@pytest.mark.parametrize("test_set", [(None, "successfully"), (MagicMock(name="blob1"), "failed")])
@patch("azure.storage.blob.ContainerClient")
def test_cloud_cache_command(mock_container_client, test_set):
    # setup
    command_args = [
        "cloud-cache",
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
    with unittest.TestCase().assertLogs(logger="olive.cli.cloud_cache", level="INFO") as log:
        cli_main(command_args)

    # assert
    assert (test_set[1] in message for message in log.output), "Expected log message not found."
    mock_container_client.assert_called_once()
    mock_container_client().delete_blob.assert_called_once()


# TODO(anyone): Add tests for ManageAMLComputeCommand
# Test for ExportAdaptersCommand is added as part of test/unit_test/passes/onnx/test_export_adapters.py
