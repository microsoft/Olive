# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Union

import paramiko

from olive.common.constants import OS
from olive.common.utils import get_path_by_os
from olive.exception import OliveSystemError
from olive.logging import WORKFLOW_COMPLETED_LOG
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

    def submit_workflow(
        self,
        olive_config: Dict,
        workflow_id: str,
        remote_cache_dir: Path,
        local_cache_dir: Path,
        remote_output_dir: Path,
        local_output_dir: Path,
    ) -> str:
        logger.info("Submitting workflow %s to the cloud system.", workflow_id)
        self._upload_config_file(workflow_id, olive_config, self.olive_path)
        self._run_workflow(workflow_id)
        self._download_workflow_output(
            self.ssh_client.open_sftp(), remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
        )

    def _connect_to_vm(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.connect(
            hostname=self.hostname, key_filename=self.key_filename, username=self.username, password=self.password
        )
        return ssh_client

    def _run_workflow(self, workflow_id: str):
        olive_workflow_cmd = f"python3 -m olive run --config {workflow_id}_config.json"

        conda_init_cmd = f"source {self.conda_path}"
        av_cmd = f"conda activate {self.conda_name}"
        cmd = f"{conda_init_cmd} && {av_cmd} && cd {self.olive_path} && {olive_workflow_cmd}"
        logger.info("Workflow %s is running in the cloud system.", workflow_id)
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

    def retrieve_workflow_logs(
        self,
        workflow_id: str,
        remote_cache_dir: Path,
        local_cache_dir: Path,
        remote_output_dir: Path,
        local_output_dir: Path,
    ):
        local_cache_dir = local_cache_dir / workflow_id
        remote_cache_dir = remote_cache_dir / workflow_id

        log_path = get_path_by_os(Path(remote_cache_dir) / f"{workflow_id}.log", self.os)
        sftp = self.ssh_client.open_sftp()
        try:
            with sftp.file(log_path, "r") as log_file:

                last_log = None

                for line in log_file:
                    log = line.strip()
                    logger.info(log)
                    last_log = log

                if WORKFLOW_COMPLETED_LOG in last_log:
                    logger.info("Workflow is completed. Downloading outputs and cache.")
                    self._download_workflow_output(
                        sftp, remote_cache_dir, local_cache_dir, remote_output_dir, local_output_dir
                    )
                else:
                    logger.warning("Workflow is still running. Please wait for completion.")
        except (OSError, FileNotFoundError):
            logger.exception(
                "Failed to retrieve workflow logs with workflow_id %s. Please run workflow first.", workflow_id
            )
        finally:
            sftp.close()

    def _download_workflow_output(
        self, sftp, remote_cache_dir: Path, local_cache_dir: Path, remote_output_dir: Path, local_output_dir: Path
    ):
        self._download_directory(sftp, remote_output_dir, local_output_dir)
        logger.info("Downloaded workflow output to %s", local_output_dir)
        self._download_directory(sftp, remote_cache_dir, local_cache_dir)
        logger.info("Downloaded workflow cache to %s", local_cache_dir)

    def _download_directory(self, sftp, remote_dir: Path, local_dir: Path):
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)

        remote_dir = get_path_by_os(remote_dir, self.os)

        for item in sftp.listdir_attr(remote_dir):
            remote_path = get_path_by_os(Path(remote_dir) / item.filename, self.os)
            local_path = local_dir / item.filename

            if stat.S_ISDIR(item.st_mode):
                self._download_directory(sftp, Path(remote_path), local_path)

            else:
                logger.info("Downloading %s to %s", remote_path, local_path)
                sftp.get(remote_path, local_path)

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
