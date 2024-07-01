# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import stat
import tempfile
from pathlib import Path
from typing import Dict, Union

import paramiko

from olive.common.config_utils import ConfigBase
from olive.common.utils import get_path_by_os
from olive.exception import OliveDispatcherError
from olive.logging import WORKFLOW_COMPLETED_LOG
from olive.workflows.dispatcher.dispatcher_config import Dispatcher, DispatcherType
from olive.workflows.run.config import RunConfig

logger = logging.getLogger(__name__)


class RemoteDispatcherConfig(ConfigBase):
    hostname: str
    workflow_path: str
    conda_path: str
    conda_name: str
    os: str
    username: str
    key_filename: str = None
    password: str = None

    @staticmethod
    def load_from_file(config_path: str):
        config_path = Path(config_path)
        with config_path.open() as f:
            return RemoteDispatcherConfig.load_from_dict(json.load(f))

    @staticmethod
    def load_from_dict(config_dict: dict):
        return RemoteDispatcherConfig(**config_dict)


class RemoteDispatcher(Dispatcher):
    dispatcher_type = DispatcherType.Remote

    def __init__(self, config_path: str) -> None:
        self.config = self.load_config(config_path)
        self.ssh_client = self._connect_to_vm()

    def load_config(self, config_path) -> RemoteDispatcherConfig:
        return RemoteDispatcherConfig.load_from_file(config_path)

    def submit_workflow(self, run_config: RunConfig):
        cache_dir = run_config.engine.cache_dir
        output_dir = run_config.engine.output_dir
        run_config.dispatcher = None
        run_config.engine.log_to_file = True
        os = self.config.os
        run_config.engine.cache_dir = get_path_by_os(Path(self.config.workflow_path) / cache_dir, os)
        run_config.engine.output_dir = get_path_by_os(Path(self.config.workflow_path) / output_dir, os)
        olive_config = run_config.to_json()
        return self._submit_workflow(olive_config, run_config.workflow_id, cache_dir, output_dir)

    def _submit_workflow(self, olive_config: Dict, workflow_id: str, cache_dir: Path, output_dir: Path):
        logger.info("Submitting workflow %s to the remote dispatcher.", workflow_id)
        self._check_workflow_path(self.config.workflow_path)
        self._upload_config_file(workflow_id, olive_config, self.config.workflow_path)
        self._run_workflow(workflow_id)
        self._download_workflow_output(self.ssh_client.open_sftp(), cache_dir, output_dir, workflow_id)

    def _connect_to_vm(self):
        ssh_client = paramiko.SSHClient()
        ssh_client.load_system_host_keys()
        ssh_client.connect(
            hostname=self.config.hostname,
            key_filename=self.config.key_filename,
            username=self.config.username,
            password=self.config.password,
        )
        return ssh_client

    def _check_workflow_path(self, workflow_path: str):
        cmd = f"if [ ! -d {workflow_path} ]; then mkdir -p {workflow_path}; fi"
        self._run_command(cmd)

    def _run_workflow(self, workflow_id: str):
        olive_workflow_cmd = f"python3 -m olive run --config {workflow_id}_config.json"

        conda_init_cmd = f"source {self.config.conda_path}"
        av_cmd = f"conda activate {self.config.conda_name}"
        cmd = f"{conda_init_cmd} && {av_cmd} && cd {self.config.workflow_path} && {olive_workflow_cmd}"
        logger.info("Workflow %s is running on the remote dispatcher.", workflow_id)
        self._run_command(cmd)

    def _upload_config_file(self, workflow_id: str, olive_config: Dict, target_path: Union[Path, str]):
        target_path = get_path_by_os(Path(target_path) / f"{workflow_id}_config.json", self.config.os)
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
        cache_dir: Path,
        output_dir: Path,
    ):
        remote_cache_dir = Path(self.config.workflow_path) / cache_dir / workflow_id
        log_path = get_path_by_os(Path(remote_cache_dir) / f"{workflow_id}.log", self.config.os)
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
                    self._download_workflow_output(sftp, cache_dir, output_dir, workflow_id)
                else:
                    logger.warning("Workflow is still running. Please wait for completion.")
        except (OSError, FileNotFoundError):
            logger.exception(
                "Failed to retrieve workflow logs with workflow_id %s. Please run workflow first.", workflow_id
            )
        finally:
            sftp.close()

    def _download_workflow_output(self, sftp, cache_dir: Path, output_dir: Path, workflow_id: str):
        remote_output_dir = Path(self.config.workflow_path) / output_dir
        logger.info("Downloading workflow output to %s", output_dir)
        self._download_directory(sftp, remote_output_dir, output_dir)

        local_cache_dir = cache_dir / workflow_id
        remote_cache_dir = Path(self.config.workflow_path) / cache_dir / workflow_id
        logger.info("Downloading workflow cache to %s", local_cache_dir)
        self._download_directory(sftp, remote_cache_dir, local_cache_dir)

    def _download_directory(self, sftp, remote_dir: Path, local_dir: Path):
        if not local_dir.exists():
            local_dir.mkdir(parents=True, exist_ok=True)

        remote_dir = get_path_by_os(remote_dir, self.config.os)

        for item in sftp.listdir_attr(remote_dir):
            remote_path = get_path_by_os(Path(remote_dir) / item.filename, self.config.os)
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
            raise OliveDispatcherError(f"Command failed with exit status {exit_status}" and error_output)
