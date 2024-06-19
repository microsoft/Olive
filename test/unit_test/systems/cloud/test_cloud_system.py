# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock, patch

from olive.common.constants import OS
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
            f"python3 -m olive.workflows.run --config {workflow_id}_config.json"
        )

        # execute
        cloud_sys.submit_workflow(olive_config, workflow_id)

        # assert
        mock_run_command.assert_called_once_with(expected_cmd)
