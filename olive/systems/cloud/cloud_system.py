# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import paramiko

from olive.common.constants import DEFAULT_WORKFLOW_ID, OS
from olive.common.utils import get_path_by_os
from olive.exception import OliveSystemError
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric
    from olive.evaluator.metric_result import MetricResult
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ModelConfig

logger = logging.getLogger(__name__)


class CloudSystem(OliveSystem):
    system_type = SystemType.Cloud

    def __init__(
        self,
        olive_path: Union[Path, str],
        conda_path: Union[Path, str],
        hostname: str,
        conda_name: str,
        os: str = OS.LINUX,
        key_filename: str = None,
        username: str = None,
        password: str = None,
        accelerators: List[AcceleratorConfig] = None,
        hf_token: bool = None,
    ):
        super().__init__(accelerators, hf_token)
        self.os = os
        self.olive_path = get_path_by_os(olive_path, os)
        self.conda_path = get_path_by_os(conda_path, os)
        self.conda_name = conda_name
        self.hostname = hostname
        self.key_filename = key_filename
        self.username = username
        self.password = password
        self.ssh_client = self._connect_to_vm()

    def submit_workflow(self, olive_config: Dict, workflow_id: str = DEFAULT_WORKFLOW_ID) -> str:
        logger.info("Workflow %s is running in the cloud system.", workflow_id)
        self._upload_config_file(workflow_id, olive_config, self.olive_path)
        self._run_workflow(workflow_id)
        # TODO(xiaoyu): add output return

    def _connect_to_vm(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        ssh_client.connect(
            hostname=self.hostname, key_filename=self.key_filename, username=self.username, password=self.password
        )
        return ssh_client

    def _run_workflow(self, workflow_id: str):
        olive_workflow_cmd = f"python3 -m olive run --config {workflow_id}_config.json"

        conda_init_cmd = f"source {self.conda_path}"
        av_cmd = f"conda activate {self.conda_name}"
        cmd = f"{conda_init_cmd} && {av_cmd} && cd {self.olive_path} && {olive_workflow_cmd}"
        self._run_command(cmd)

    def _upload_config_file(self, workflow_id: str, olive_config: Dict, target_path: Union[Path, str]):
        target_path = get_path_by_os(Path(target_path) / f"{workflow_id}_config.json", self.os)
        with tempfile.TemporaryDirectory() as temp_dir:
            olive_config_path = Path(temp_dir) / "config.json"
            with open(olive_config_path, "w") as fout:
                json.dump(olive_config, fout)

            sftp = self.ssh_client.open_sftp()
            sftp.put(str(olive_config_path), target_path)
            logger.debug("Uploaded config file to %s", target_path)

    def _run_command(self, command: str):
        logger.info("Running command %s", command)
        _, stdout, stderr = self.ssh_client.exec_command(command)
        logger.info("Output:")
        for line in iter(stdout.readline, ""):
            logger.info(line)

        error_output = stderr.read().decode()
        if error_output:
            logger.error("\nError:")
            logger.error(error_output)

        exit_status = stdout.channel.recv_exit_status()
        if exit_status == 0:
            logger.info("\nCommand executed successfully")
            return 0
        else:
            raise OliveSystemError(f"Command failed with exit status {exit_status}" and error_output)

    def evaluate_model(
        self, model_config: "ModelConfig", data_root: str, metrics: List["Metric"], accelerator: "AcceleratorSpec"
    ) -> "MetricResult":
        """Evaluate the model."""
        raise NotImplementedError

    def remove(self):
        raise NotImplementedError

    def run_pass(self, the_pass, model_config, data_root, output_model_path, point=None):
        raise NotImplementedError
