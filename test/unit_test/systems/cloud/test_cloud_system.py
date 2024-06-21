# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import stat
from pathlib import Path
from unittest.mock import MagicMock, patch

from olive.common.constants import OS
from olive.logging import WORKFLOW_COMPLETED_LOG
from olive.systems.cloud.cloud_system import CloudSystem

# pylint: disable=W0212


class TestCloudSystem:

    OLIVE_PATH = "/home/ucm/olive"
    CONDA_PATH = "/home/ucm/conda"
    HOSTNAME = "hostname"
    CONDA_NAME = "olive"
    USERNAME = "ucm"
    KEY_FILENAME = "/home/ucm/.ssh/key.pem"

    @patch("paramiko.SSHClient")
    def test__run_command(self, mock_ssh_client):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        command = "pwd"
        mock_stdout = MagicMock()
        mock_stderr = MagicMock()
        mock_stdout.readline.side_effect = ["stdout", ""]
        mock_stderr.read.decode.return_value = None
        mock_stdout.channel.recv_exit_status.return_value = 0
        cloud_sys.ssh_client.exec_command.return_value = ("stdin", mock_stdout, mock_stderr)

        # execute
        result = cloud_sys._run_command(command)

        # assert
        assert result == 0

    @patch("paramiko.SSHClient")
    def test__upload_config_file(self, mock_ssh_client):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        workflow_id = "test_workflow"
        olive_config = {"key": "value"}
        mock_sftp = MagicMock()
        cloud_sys.ssh_client.open_sftp.return_value = mock_sftp

        # execute
        cloud_sys._upload_config_file(workflow_id, olive_config, cloud_sys.olive_path)

        # assert
        mock_sftp.put.assert_called_once()

    @patch("olive.systems.cloud.CloudSystem._upload_config_file")
    @patch("olive.systems.cloud.CloudSystem._run_command")
    @patch("paramiko.SSHClient")
    def test_submit_workflow(self, mock_ssh_client, mock_run_command, mock_upload_config_file):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        olive_config = {"key": "value"}
        workflow_id = "test_workflow"
        expected_cmd = (
            f"source {self.CONDA_PATH} && "
            f"conda activate {self.CONDA_NAME} && "
            f"cd {self.OLIVE_PATH} && "
            f"python3 -m olive run --config {workflow_id}_config.json"
        )
        workflow_id = "test_workflow"
        remote_cache_dir = Path("/remote/cache")
        local_cache_dir = Path("/local/cache")
        remote_output_dir = Path("/remote/output")
        local_output_dir = Path("/local/output")

        # execute
        cloud_sys.submit_workflow(
            olive_config, workflow_id, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

        # assert
        mock_run_command.assert_called_once_with(expected_cmd)

    @patch("olive.systems.cloud.CloudSystem._download_workflow_output")
    @patch("paramiko.SSHClient")
    def test_retrieve_workflow_logs_workflow_completed(self, mock_ssh_client, mock_download_workflow_output):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        workflow_id = "test_workflow"
        remote_cache_dir = Path("/remote/cache")
        local_cache_dir = Path("/local/cache")
        remote_output_dir = Path("/remote/output")
        local_output_dir = Path("/local/output")
        mock_sftp = MagicMock()
        cloud_sys.ssh_client.open_sftp.return_value = mock_sftp
        mock_file = MagicMock()
        mock_sftp.file.return_value = mock_file
        mock_file.__enter__.return_value = iter([WORKFLOW_COMPLETED_LOG])

        # execute
        cloud_sys.retrieve_workflow_logs(
            workflow_id, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

        # assert
        mock_download_workflow_output.assert_called_once_with(
            mock_sftp, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

    @patch("olive.systems.cloud.CloudSystem._download_workflow_output")
    @patch("olive.systems.cloud.cloud_system.logger")
    @patch("paramiko.SSHClient")
    def test_retrieve_workflow_logs_workflow_running(self, mock_ssh_client, mock_logger, mock_download_workflow_output):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        workflow_id = "test_workflow"
        remote_cache_dir = Path("/remote/cache")
        local_cache_dir = Path("/local/cache")
        remote_output_dir = Path("/remote/output")
        local_output_dir = Path("/local/output")
        mock_sftp = MagicMock()
        cloud_sys.ssh_client.open_sftp.return_value = mock_sftp
        mock_file = MagicMock()
        mock_sftp.file.return_value = mock_file
        mock_file.__enter__.return_value = iter(["Workflow is running."])

        # execute
        cloud_sys.retrieve_workflow_logs(
            workflow_id, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

        # assert
        mock_logger.warning.assert_called_once_with("Workflow is still running. Please wait for completion.")
        mock_download_workflow_output.assert_not_called()

    @patch("olive.systems.cloud.CloudSystem._download_directory")
    @patch("paramiko.SSHClient")
    def test__download_workflow_output(self, mock_ssh_client, mock_download_directory):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        mock_sftp = MagicMock()
        cloud_sys.ssh_client.open_sftp.return_value = mock_sftp
        remote_cache_dir = Path("/remote/cache")
        local_cache_dir = Path("/local/cache")
        remote_output_dir = Path("/remote/output")
        local_output_dir = Path("/local/output")

        # execute
        cloud_sys._download_workflow_output(
            mock_sftp, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

        # assert
        mock_download_directory.assert_any_call(mock_sftp, remote_output_dir, local_output_dir)
        mock_download_directory.assert_any_call(mock_sftp, remote_cache_dir, local_cache_dir)

    @patch("paramiko.SSHClient")
    def test__download_directory(self, mock_ssh_client):
        # setup
        cloud_sys = CloudSystem(
            self.OLIVE_PATH,
            self.CONDA_PATH,
            self.HOSTNAME,
            self.CONDA_NAME,
            OS.LINUX,
            self.KEY_FILENAME,
            self.USERNAME,
        )
        remote_dir = Path("/remote/dir")
        local_dir = Path("/local/dir")
        mock_sftp = MagicMock()
        cloud_sys.ssh_client.open_sftp.return_value = mock_sftp
        mock_sftp.listdir_attr.return_value = [
            MagicMock(st_mode=stat.S_IFREG, filename="file"),
        ]

        # execute
        with patch.object(Path, "mkdir") as mock_mkdir, patch.object(Path, "exists") as mock_exists:
            mock_exists.return_value = False
            cloud_sys._download_directory(mock_sftp, remote_dir, local_dir)

            # assert
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)
            mock_sftp.get.assert_called_once_with("/remote/dir/file", local_dir / "file")
