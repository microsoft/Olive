# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import json
import logging
import sys
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import docker
from docker.errors import BuildError, ContainerError

from olive.common.utils import set_nested_dict_value
from olive.hardware import Device
from olive.resource_path import find_all_resources
from olive.systems.common import AcceleratorConfig, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import LocalTargetUserConfig, SystemConfig
from olive.workflows.run.config import RunConfig

if TYPE_CHECKING:
    from olive.evaluator.metric_result import MetricResult
    from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ModelConfig
    from olive.passes import Pass

logger = logging.getLogger(__name__)


class DockerSystem(OliveSystem):
    system_type = SystemType.Docker

    def __init__(
        self,
        image_name: str,
        build_context_path: str,
        dockerfile: str,
        work_dir: str,
        build_args: dict = None,
        run_params: dict = None,
        clean_image: bool = True,
        accelerators: list[AcceleratorConfig] = None,
        hf_token: bool = None,
        **kwargs,  # used to hold the rest of the arguments not used by dockersystem.
    ):
        super().__init__(accelerators=accelerators, hf_token=hf_token)

        logger.info("Initializing Docker System...")
        self.docker_client = docker.from_env()

        if not build_context_path and not dockerfile:
            raise ValueError("build_context_path and dockerfile must be provided.")

        self.image_name = image_name
        self.build_context_path = build_context_path
        self.dockerfile = dockerfile
        self.build_args = build_args or {}
        self.run_params = run_params or {}
        self.work_dir = Path(work_dir)
        self.clean_image = clean_image

        # Get or build Docker image
        self.image = self._get_or_build_image()

    def _get_or_build_image(self):
        """Get existing Docker image or build a new one."""
        try:
            image = self.docker_client.images.get(self.image_name)
            logger.info("Image %s found", self.image_name)
            return image
        except docker.errors.ImageNotFound:
            return self._build_image()

    def _build_image(self):
        """Build Docker image with real-time logging."""
        dockerfile_path = Path(self.build_context_path) / self.dockerfile
        logger.info(
            "Building image from Dockerfile %s with buildargs %s",
            dockerfile_path,
            self.build_args,
        )

        logger.info("Building Docker image %s...", self.image_name)
        for log in self.docker_client.api.build(
            path=self.build_context_path,
            dockerfile=self.dockerfile,
            tag=self.image_name,
            buildargs=self.build_args,
            decode=True,
        ):
            if "stream" in log:
                line = log["stream"].strip()
                if line:
                    logger.info("[Docker Build] %s", line)
                    sys.stdout.flush()
            elif "error" in log:
                logger.error("[Docker Build Error] %s", log["error"])
                raise BuildError(log["error"])

        image = self.docker_client.images.get(self.image_name)
        logger.info("Image %s built successfully.", self.image_name)
        return image

    def run_workflow(self, run_config: RunConfig):
        """Run Olive workflow in Docker container."""
        with tempfile.TemporaryDirectory() as tempdir:
            self._run_workflow(run_config, tempdir)

    def _run_workflow(self, run_config: RunConfig, tmp_dir: str):
        """Execute workflow in Docker container."""
        volumes = []

        # Mount runner script
        runner_path, runner_mount = self._create_runner_script_mount()
        volumes.append(runner_mount)

        # Prepare container configuration
        container_config = self._prepare_container_config(run_config)

        # Mount workflow resources
        container_config = self._mount_workflow_config(container_config, volumes)

        # Create and mount config file
        config_path, config_mount = self._create_config_mount(tmp_dir, container_config)
        volumes.append(config_mount)

        # Build command
        command = f"python {runner_path} --config {config_path}"

        # Get device from accelerators
        device = self._get_device(run_config)

        # Run container
        self._run_container(command, volumes, device=device)

        # Clean up if requested
        if self.clean_image:
            self.remove()

    def _prepare_container_config(self, run_config: RunConfig) -> dict:
        """Prepare run config for container environment."""
        # Replace docker system with local system
        accelerators = run_config.engine.host.config.accelerators
        run_config.engine.host = SystemConfig(
            type=SystemType.Local, config=LocalTargetUserConfig(accelerators=accelerators)
        )

        if run_config.engine.target.type not in (SystemType.Local, SystemType.PythonEnvironment):
            raise ValueError(
                "Docker system does not support target system other than LocalSystem or PythonEnvironment."
            )

        # Convert to JSON and remove unnecessary fields
        config = run_config.to_json(make_absolute=False)
        config.pop("evaluators", None)
        return config

    def _get_device(self, run_config: RunConfig) -> Optional[Device]:
        """Extract device from run config."""
        accelerators = run_config.engine.host.config.accelerators
        return accelerators[0].device if accelerators else None

    def _mount_workflow_config(
        self,
        config: dict,
        volumes: list[str],
        ignore_keys: Optional[list[str]] = None,
    ) -> dict:
        """Mount workflow resources and update config paths."""
        config_copy = copy.deepcopy(config)
        ignore_keys = ignore_keys or ["systems", "evaluators"]

        # Find and mount all resources
        all_resources = find_all_resources(config_copy, ignore_keys=ignore_keys)
        for resource_key, resource_path in all_resources.items():
            # Create container path
            container_path = str(self.work_dir / Path(resource_path.get_path()).name)

            # Add volume mount
            mount_string = f"{resource_path.get_path()}:{container_path}"
            volumes.append(mount_string)

            # Update config with container path
            set_nested_dict_value(config_copy, resource_key, container_path)

        logger.debug("Container config: %s", config_copy)
        return config_copy

    def _run_container(
        self,
        command: str,
        volumes: list[str],
        device: Optional[Device] = None,
    ):
        """Run Docker container with given command and volumes."""
        # Prepare run parameters
        run_params = self._prepare_run_params()

        # Prepare environment variables
        environment = self._prepare_environment(run_params.pop("environment", {}))

        # Add GPU support if needed
        if device == Device.GPU:
            run_params["device_requests"] = [docker.types.DeviceRequest(capabilities=[["gpu"]])]

        # Run container
        logger.info("Container is running, it will take a while...")
        logger.debug("Running container with command: %s", command)

        container = self.docker_client.containers.run(
            image=self.image,
            command=command,
            volumes=volumes,
            detach=True,
            environment=environment,
            **run_params,
        )

        # Stream logs and check result
        self._stream_and_check_container(container, command)

    def _prepare_run_params(self) -> dict:
        """Convert run params for Docker API."""
        if not self.run_params:
            return {}
        # Convert hyphenated keys to underscores
        return {k.replace("-", "_"): v for k, v in self.run_params.items()}

    def _prepare_environment(self, base_env) -> dict:
        """Prepare environment variables for container."""
        # Convert list to dict if needed
        if isinstance(base_env, list):
            environment = {env.split("=")[0]: env.split("=")[1] for env in base_env}
        else:
            environment = base_env.copy() if isinstance(base_env, dict) else {}

        # Add default environment variables
        environment.setdefault("PYTHONPYCACHEPREFIX", "/tmp")
        environment["OLIVE_LOG_LEVEL"] = logging.getLevelName(logger.getEffectiveLevel())

        # Add HuggingFace token if needed
        if self.hf_token:
            token = self._get_huggingface_token()
            if token:
                environment["HF_TOKEN"] = token

        return environment

    def _stream_and_check_container(self, container, command: str):
        """Stream container logs and check exit status."""
        # Stream logs
        for line in container.logs(stream=True, follow=True):
            log = line.decode().strip()
            if log:
                logger.info("[Docker] %s", log)
                sys.stdout.flush()

        # Check exit code
        exit_code = container.wait()["StatusCode"]
        container.remove()

        if exit_code != 0:
            raise ContainerError(
                container,
                exit_code,
                command,
                self.image,
                "Docker container run failed. Please check the logs for more details.",
            )

        logger.debug("Docker container run completed successfully")

    def remove(self):
        """Remove Docker image."""
        image_ref = self.image.tags[0] if self.image.tags else self.image.id
        self.docker_client.images.remove(image_ref, force=True)
        logger.info("Image %s removed successfully.", image_ref)

    def _create_config_mount(self, tmp_dir: str, config: dict) -> tuple[str, str]:
        """Create mount for config file."""
        # Save config to temporary file
        config_file_path = Path(tmp_dir) / "config.json"
        with config_file_path.open("w") as f:
            json.dump(config, f, indent=4)

        # Create mount string
        container_path = str(self.work_dir / "config.json")
        mount_string = f"{config_file_path}:{container_path}"
        return container_path, mount_string

    def _create_runner_script_mount(self) -> tuple[str, str]:
        """Create mount for runner script."""
        runner_script_name = "workflow_runner.py"
        container_path = str(self.work_dir / runner_script_name)
        host_path = Path(__file__).resolve().parent / runner_script_name
        mount_string = f"{host_path}:{container_path}"
        return container_path, mount_string

    @staticmethod
    def _get_huggingface_token() -> Optional[str]:
        """Get HuggingFace token from environment or file."""
        import os

        # Check environment variable
        token = os.getenv("HF_TOKEN")
        if token:
            return token

        # Check token file
        token_path = Path.home() / ".huggingface" / "token"
        if token_path.exists():
            with token_path.open() as f:
                return f.read().strip()

        logger.error(
            "HuggingFace token is required but not found. "
            "Please set HF_TOKEN environment variable or login using 'huggingface-cli login'."
        )
        return None

    def run_pass(self, the_pass: "Pass", model_config: "ModelConfig", output_model_path: str) -> "ModelConfig":
        raise NotImplementedError("DockerSystem does not support run_pass. Use run_workflow instead.")

    def evaluate_model(
        self, model_config: "ModelConfig", evaluator_config: "OliveEvaluatorConfig", accelerator: "AcceleratorSpec"
    ) -> "MetricResult":
        raise NotImplementedError("DockerSystem does not support evaluate_model. Use run_workflow instead.")
