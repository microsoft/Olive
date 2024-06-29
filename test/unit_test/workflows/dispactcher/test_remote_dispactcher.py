# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import stat
from pathlib import Path
from unittest.mock import MagicMock, call, patch

from olive.common.constants import OS
from olive.logging import WORKFLOW_COMPLETED_LOG
from olive.workflows.dispactcher.remote_dispactcher import RemoteDispactcher

# pylint: disable=W0212


class TestRemoteDispactcher:
    

    @patch("paramiko.SSHClient")
    def test__run_command(self, mock_ssh_client):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))
        command = "pwd"
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.readline.side_effect = ["stdout", ""]
        mock_stderr.read.decode.return_value = None
        mock_stdout.channel.recv_exit_status.return_value = 0
        remote_dispactcher.ssh_client.exec_command.return_value = ("stdin", mock_stdout, mock_stderr)

        # execute
        result = remote_dispactcher._run_command(command)

        # assert
        assert result == 0

    @patch("paramiko.SSHClient")
    def test__upload_config_file(self, mock_ssh_client):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))
        workflow_id = "test_workflow"
        olive_config = {"key": "value"}
        mock_sftp = MagicMock()
        remote_dispactcher.ssh_client.open_sftp.return_value = mock_sftp

        # execute
        remote_dispactcher._upload_config_file(workflow_id, olive_config, remote_dispactcher.config.workflow_path)

        # assert
        mock_sftp.put.assert_called_once()

    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._download_workflow_output")
    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._upload_config_file")
    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._run_command")
    @patch("paramiko.SSHClient")
    def test__submit_workflow(
        self, mock_ssh_client, mock_run_command, mock_upload_config_file, mock_download_workflow_output
    ):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))
        olive_config = {"key": "value"}
        workflow_id = "test_workflow"
        expected_cmd_1 = f"if [ ! -d {remote_dispactcher.config.workflow_path} ]; then mkdir -p {remote_dispactcher.config.workflow_path}; fi"
        expected_cmd_2 = (
            f"source {remote_dispactcher.config.conda_path} && "
            f"conda activate {remote_dispactcher.config.conda_name} && "
            f"cd {remote_dispactcher.config.workflow_path} && "
            f"python3 -m olive run --config {workflow_id}_config.json"
        )
        workflow_id = "test_workflow"
        cache_dir = Path("cache")
        output_dir = Path("output")
        expected_calls = [expected_cmd_1, expected_cmd_2]

        # execute
        remote_dispactcher._submit_workflow(olive_config, workflow_id, cache_dir, output_dir)

        # assert
        mock_upload_config_file.assert_called_once_with(workflow_id, olive_config, remote_dispactcher.config.workflow_path)
        mock_run_command.assert_has_calls([call(cmd) for cmd in expected_calls])
        mock_download_workflow_output.assert_called_once()

    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._download_workflow_output")
    @patch("paramiko.SSHClient")
    def test_retrieve_workflow_logs_workflow_completed(self, mock_ssh_client, mock_download_workflow_output):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))
        workflow_id = "test_workflow"
        cache_dir = Path("cache")
        output_dir = Path("output")
        mock_sftp = MagicMock()
        remote_dispactcher.ssh_client.open_sftp.return_value = mock_sftp
        mock_file = MagicMock()
        mock_sftp.file.return_value = mock_file
        mock_file.__enter__.return_value = iter([WORKFLOW_COMPLETED_LOG])

        # execute
        remote_dispactcher.retrieve_workflow_logs(workflow_id, cache_dir, output_dir)

        # assert
        mock_download_workflow_output.assert_called_once_with(mock_sftp, cache_dir, output_dir, workflow_id)

    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._download_workflow_output")
    @patch("olive.workflows.dispactcher.remote_dispactcher.logger")
    @patch("paramiko.SSHClient")
    def test_retrieve_workflow_logs_workflow_running(self, mock_ssh_client, mock_logger, mock_download_workflow_output):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))

        workflow_id = "test_workflow"
        cache_dir = Path("/remote/cache")
        output_dir = Path("/remote/output")
        mock_sftp = MagicMock()
        remote_dispactcher.ssh_client.open_sftp.return_value = mock_sftp
        mock_file = MagicMock()
        mock_sftp.file.return_value = mock_file
        mock_file.__enter__.return_value = iter(["Workflow is running."])

        # execute
        remote_dispactcher.retrieve_workflow_logs(workflow_id, cache_dir, output_dir)

        # assert
        mock_logger.warning.assert_called_once_with("Workflow is still running. Please wait for completion.")
        mock_download_workflow_output.assert_not_called()

    @patch("olive.workflows.dispactcher.remote_dispactcher.RemoteDispactcher._download_directory")
    @patch("paramiko.SSHClient")
    def test__download_workflow_output(self, mock_ssh_client, mock_download_directory):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))

        workflow_id = "test_workflow"
        mock_sftp = MagicMock()
        remote_dispactcher.ssh_client.open_sftp.return_value = mock_sftp
        cache_dir = Path("cache")
        output_dir = Path("output")
        local_output_dir = output_dir
        remote_output_dir = Path(remote_dispactcher.config.workflow_path) / output_dir
        local_cache_dir = cache_dir / workflow_id
        remote_cache_dir = Path(remote_dispactcher.config.workflow_path) / cache_dir / workflow_id

        # execute
        remote_dispactcher._download_workflow_output(
            mock_sftp, cache_dir, output_dir, workflow_id
        )

        # assert
        mock_download_directory.assert_any_call(mock_sftp, remote_output_dir, local_output_dir)
        mock_download_directory.assert_any_call(mock_sftp, remote_cache_dir, local_cache_dir)

    @patch("paramiko.SSHClient")
    def test__download_directory(self, mock_ssh_client):
        # setup
        remote_dispactcher = RemoteDispactcher(str(Path(__file__).parent / "remote_dispactcher_config.json"))

        remote_dir = Path("/remote/dir")
        local_dir = Path("/local/dir")
        mock_sftp = MagicMock()
        remote_dispactcher.ssh_client.open_sftp.return_value = mock_sftp
        mock_sftp.listdir_attr.return_value = [
            MagicMock(st_mode=stat.S_IFREG, filename="file"),
        ]

        # execute
        with patch.object(Path, "mkdir") as mock_mkdir, patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            remote_dispactcher._download_directory(mock_sftp, remote_dir, local_dir)

            # assert
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_sftp.get.assert_called_once_with("/remote/dir/file", local_dir / "file")
