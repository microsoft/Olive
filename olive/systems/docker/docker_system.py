# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import json
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import docker

import olive.systems.docker.utils as docker_utils
from olive.common.config_utils import validate_config
from olive.evaluator.metric import Metric, MetricResult
from olive.hardware import Device
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig
from olive.passes import Pass
from olive.systems.common import LocalDockerConfig, SystemType
from olive.systems.olive_system import OliveSystem

logger = logging.getLogger(__name__)


class DockerSystem(OliveSystem):
    system_type = SystemType.Docker

    BASE_DOCKERFILE = "Dockerfile"

    def __init__(
        self,
        local_docker_config: Union[Dict[str, Any], LocalDockerConfig] = None,
        accelerators: List[str] = None,
        is_dev: bool = False,
        olive_managed_env: bool = False,
        requirements_file: Union[Path, str] = None,
    ):
        super().__init__(accelerators=accelerators, olive_managed_env=olive_managed_env)
        logger.info("Initializing Docker System...")
        self.is_dev = is_dev
        self.requirements_file = requirements_file
        if local_docker_config and olive_managed_env:
            raise ValueError("Olive managed environment is not supported if local_docker_config is provided.")
        if local_docker_config:
            self.docker_client = docker.from_env()
            local_docker_config = validate_config(local_docker_config, LocalDockerConfig)
            self.run_params = local_docker_config.run_params
            try:
                self.image = self.docker_client.images.get(local_docker_config.image_name)
                logger.info(f"Image {local_docker_config.image_name} found")

            except docker.errors.ImageNotFound:
                if local_docker_config.build_context_path and local_docker_config.dockerfile:
                    build_context_path = local_docker_config.build_context_path
                    logger.info(f"Building image from Dockerfile {build_context_path}/{local_docker_config.dockerfile}")
                    self.image = self.docker_client.images.build(
                        path=build_context_path,
                        dockerfile=local_docker_config.dockerfile,
                        tag=local_docker_config.image_name,
                        buildargs=local_docker_config.build_args,
                    )[0]
                elif local_docker_config.requirements_file_path:
                    logger.info(
                        f"Building image from Olive default Dockerfile with buildargs {local_docker_config.build_args} "
                        f"requirements.txt {local_docker_config.requirements_file_path}"
                    )
                    dockerfile_path = str(Path(__file__).resolve().parent / self.BASE_DOCKERFILE)
                    with tempfile.TemporaryDirectory() as tempdir:
                        build_context_path = tempdir
                        shutil.copy2(dockerfile_path, build_context_path)
                        shutil.copy2(local_docker_config.requirements_file_path, build_context_path)
                        self.image = self.docker_client.images.build(
                            path=build_context_path,
                            dockerfile=self.BASE_DOCKERFILE,
                            tag=local_docker_config.image_name,
                            buildargs=local_docker_config.build_args,
                        )[0]
                logger.info(f"Image {local_docker_config.image_name} build successfully.")

    def run_pass(
        self,
        the_pass: Pass,
        model: ModelConfig,
        data_root: str,
        output_model_path: str,
        point: Optional[Dict[str, Any]] = None,
    ) -> ModelConfig:
        """
        Run the pass on the model at a specific point in the search space.
        """
        logger.warning("DockerSystem.run_pass is not implemented yet.")
        raise NotImplementedError()

    def evaluate_model(
        self, model_config: ModelConfig, data_root: str, metrics: List[Metric], accelerator: AcceleratorSpec
    ) -> Dict[str, Any]:
        container_root_path = Path("/olive-ws/")
        with tempfile.TemporaryDirectory() as tempdir:
            metrics_res = None
            metric_json = self._run_container(
                tempdir, model_config, data_root, metrics, accelerator, container_root_path
            )
            if metric_json.is_file():
                with metric_json.open() as f:
                    metrics_res = json.load(f)
            return MetricResult.parse_obj(metrics_res)

    def _run_container(
        self,
        tempdir,
        model_config: ModelConfig,
        data_root: str,
        metrics: List[Metric],
        accelerator: AcceleratorSpec,
        container_root_path: Path,
    ):
        eval_output_path = "eval_output"
        eval_output_name = "eval_res.json"

        volumes_list = []
        eval_file_mount_path, eval_file_mount_str = docker_utils.create_eval_script_mount(container_root_path)
        volumes_list.append(eval_file_mount_str)

        if self.is_dev:
            dev_mount_path, dev_mount_str = docker_utils.create_dev_mount(tempdir, container_root_path)
            volumes_list.append(dev_mount_str)

        model_config_copy = copy.deepcopy(model_config)
        model_mounts, model_mount_str_list = docker_utils.create_model_mount(
            model_config=model_config_copy, container_root_path=container_root_path
        )
        volumes_list += model_mount_str_list

        metrics_copy = copy.deepcopy(metrics)
        volumes_list = docker_utils.create_metric_volumes_list(
            data_root=data_root,
            metrics=metrics_copy,
            container_root_path=container_root_path,
            mount_list=volumes_list,
        )

        config_mount_path, config_file_mount_str = docker_utils.create_config_file(
            tempdir=tempdir,
            model_config=model_config_copy,
            metrics=metrics_copy,
            container_root_path=container_root_path,
            model_mounts=model_mounts,
        )
        volumes_list.append(config_file_mount_str)

        output_local_path, output_mount_path, output_mount_str = docker_utils.create_output_mount(
            tempdir=tempdir,
            docker_eval_output_path=eval_output_path,
            container_root_path=container_root_path,
        )
        volumes_list.append(output_mount_str)
        logger.debug(f"The volumes list is {volumes_list}")

        eval_command = docker_utils.create_evaluate_command(
            eval_script_path=eval_file_mount_path,
            config_path=config_mount_path,
            output_path=output_mount_path,
            output_name=eval_output_name,
            accelerator=accelerator,
        )

        run_command = docker_utils.create_run_command(run_params=self.run_params)

        environment = run_command.pop("environment", {})
        envs_dict = {"PYTHONPYCACHEPREFIX": "/tmp"}
        for k, v in envs_dict.items():
            if isinstance(environment, list):
                environment = {env.split("=")[0]: env.split("=")[1] for env in environment}
            elif isinstance(environment, dict) and not environment.get(k):
                environment[k] = v

        logger.debug(f"Running container with eval command: {eval_command}")
        if accelerator.accelerator_type == Device.GPU:
            container = self.docker_client.containers.run(
                image=self.image,
                command=eval_command,
                volumes=volumes_list,
                detach=True,
                environment=environment,
                device_requests=[docker.types.DeviceRequest(capabilities=[["gpu"]])],
                **run_command,
            )
        else:
            container = self.docker_client.containers.run(
                image=self.image,
                command=eval_command,
                volumes=volumes_list,
                detach=True,
                environment=environment,
                **run_command,
            )
        docker_logs = []
        for line in container.logs(stream=True):
            # containers.logs can accept stdout/stderr as arguments, but it doesn't work
            # as we cannot ensure that all the logs will be printed in the correct channel(out/err)
            # so, we collect all the logs and print them in the end if there is an error.
            log = line.decode().strip()
            logger.debug(log)
            docker_logs.append(log)
        exit_code = container.wait()["StatusCode"]
        container.remove()
        if exit_code != 0:
            error_msg = "\n".join(docker_logs)
            raise docker.errors.ContainerError(
                container, exit_code, eval_command, self.image, f"Docker container evaluation failed with: {error_msg}"
            )
        logger.debug("Docker container evaluation completed successfully")

        metric_json = Path(output_local_path) / f"{eval_output_name}"
        return metric_json

    def remove(self):
        self.docker_client.images.remove(self.image.tags[0], force=True)
        logger.info(f"Image {self.image.tags[0]} removed successfully.")
