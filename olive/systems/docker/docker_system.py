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
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Union

import docker
from docker.errors import BuildError, ContainerError

import olive.systems.docker.utils as docker_utils
from olive.common.config_utils import ParamCategory, validate_config
from olive.evaluator.metric_result import MetricResult
from olive.hardware import Device
from olive.model import ModelConfig
from olive.systems.common import AcceleratorConfig, LocalDockerConfig, SystemType
from olive.systems.olive_system import OliveSystem
from olive.systems.system_config import DockerTargetUserConfig

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric
    from olive.evaluator.olive_evaluator import OliveEvaluatorConfig
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.passes import Pass

logger = logging.getLogger(__name__)


class DockerSystem(OliveSystem):
    system_type = SystemType.Docker

    BASE_DOCKERFILE = "Dockerfile"

    def __init__(
        self,
        local_docker_config: Union[Dict[str, Any], LocalDockerConfig],
        accelerators: List[AcceleratorConfig] = None,
        is_dev: bool = False,
        hf_token: bool = None,
        requirements_file: Optional[Union[Path, str]] = None,
        **kwargs,  # used to hold the rest of the arguments not used by dockersystem.
    ):
        super().__init__(accelerators=accelerators, hf_token=hf_token)

        logger.info("Initializing Docker System...")
        self.is_dev = is_dev
        self.docker_client = docker.from_env()
        if local_docker_config is None:
            raise ValueError("local_docker_config cannot be None.")

        local_docker_config = validate_config(local_docker_config, LocalDockerConfig)
        if not local_docker_config.build_context_path and not local_docker_config.dockerfile and not requirements_file:
            raise ValueError("build_context_path, dockerfile and requirements_file cannot be None at the same time.")

        self.config = DockerTargetUserConfig(**locals(), **kwargs)

        self.run_params = local_docker_config.run_params
        try:
            self.image = self.docker_client.images.get(local_docker_config.image_name)
            logger.info("Image %s found", local_docker_config.image_name)

        except docker.errors.ImageNotFound:
            with tempfile.TemporaryDirectory() as tempdir:
                build_context_path = tempdir
                if local_docker_config.build_context_path and local_docker_config.dockerfile:
                    dockerfile = local_docker_config.dockerfile
                    dockerfile_path = Path(local_docker_config.build_context_path) / dockerfile
                    shutil.copytree(local_docker_config.build_context_path, build_context_path, dirs_exist_ok=True)
                else:
                    dockerfile = self.BASE_DOCKERFILE
                    dockerfile_path = Path(__file__).resolve().parent / self.BASE_DOCKERFILE
                    shutil.copy2(dockerfile_path, build_context_path)

                if requirements_file:
                    shutil.copyfile(requirements_file, Path(build_context_path) / "requirements.txt")
                else:
                    requirements_dest = Path(build_context_path) / "requirements.txt"
                    if not requirements_dest.exists():
                        with (Path(build_context_path) / "requirements.txt").open("w"):
                            pass

                logger.info(
                    "Building image from Dockerfile %s with buildargs %s ",
                    dockerfile_path,
                    local_docker_config.build_args,
                )
                try:
                    self.image, build_logs = self.docker_client.images.build(
                        path=build_context_path,
                        dockerfile=dockerfile,
                        tag=local_docker_config.image_name,
                        buildargs=local_docker_config.build_args,
                    )
                    logger.info("Image %s build successfully.", local_docker_config.image_name)
                    _print_docker_logs(build_logs, logging.DEBUG)
                except BuildError as e:
                    logger.exception("Image build failed with error.")
                    _print_docker_logs(e.build_log, logging.ERROR)
                    raise

    def run_pass(self, the_pass: "Pass", model_config: "ModelConfig", output_model_path: str) -> "ModelConfig":
        """Run the pass on the model."""
        with tempfile.TemporaryDirectory() as tempdir:
            return self._run_pass_container(Path(tempdir), the_pass, model_config, output_model_path)

    def _run_pass_container(
        self, workdir: Path, the_pass: "Pass", model_config: "ModelConfig", output_model_path: str
    ) -> "ModelConfig":
        pass_config = the_pass.to_json(check_object=True)

        volumes_list = []
        runner_output_path = "runner_output"
        runner_output_name = "runner_res.json"
        container_root_path = Path("/olive-ws/")
        # mount pass_runner script
        docker_runner_path, pass_runner_file_mount_str = docker_utils.create_runner_script_mount(container_root_path)
        volumes_list.append(pass_runner_file_mount_str)

        # mount dev stuff
        if self.is_dev:
            _, dev_mount_str = docker_utils.create_dev_mount(workdir, container_root_path)
            volumes_list.append(dev_mount_str)

        # mount model
        docker_model_files, model_mount_str_list, mount_model_to_local = docker_utils.create_model_mount(
            model_config=model_config, container_root_path=container_root_path
        )
        volumes_list.extend(model_mount_str_list)

        # data_dir or data_config
        docker_data_paths, data_mount_str_list = self._create_data_mounts_for_pass(container_root_path, the_pass)
        volumes_list.extend(data_mount_str_list)

        # mount config file
        data = self._create_runner_config(model_config, pass_config, docker_model_files, docker_data_paths)
        logger.debug("Runner config is %s", data)
        docker_config_file, config_file_mount_str = docker_utils.create_config_file(
            workdir=workdir,
            config=data,
            container_root_path=container_root_path,
        )
        volumes_list.append(config_file_mount_str)

        # output mount
        output_local_path, docker_output_path, output_mount_str = docker_utils.create_output_mount(
            workdir=workdir,
            docker_output_path=runner_output_path,
            container_root_path=container_root_path,
        )
        volumes_list.append(output_mount_str)
        logger.debug("The volumes list is %s", volumes_list)

        runner_command = docker_utils.create_runner_command(
            runner_script_path=docker_runner_path,
            config_path=docker_config_file,
            output_path=docker_output_path,
            output_name=runner_output_name,
        )

        model_output_json_file = self._run_container(
            runner_command, volumes_list, output_local_path, runner_output_name, the_pass.accelerator_spec
        )
        if model_output_json_file.is_file():
            with model_output_json_file.open() as f:
                model_output = json.load(f)
                output_model = ModelConfig.parse_obj(model_output)
                logger.debug("Copying model from %s to %s", output_local_path, output_model_path)
                shutil.copytree(output_local_path, output_model_path, dirs_exist_ok=True)
                logger.debug("mount_model_to_local: %s", mount_model_to_local)
                for resource_name, resource_str in output_model.get_resource_strings().items():
                    if not resource_str:
                        continue

                    logger.debug("Resource %s path: %s", resource_name, resource_str)
                    original_resource_path = mount_model_to_local.get(resource_str)
                    if original_resource_path:
                        # If the output model path is something like /olive-ws/model.onnx
                        # we need replace with the original model path
                        output_model.config[resource_name] = original_resource_path
                        logger.info("Original resource path for %s is: %s", resource_str, original_resource_path)
                        continue

                    # output_local_path should be something like: /tmp/tmpd1sjw9xa/runner_output
                    # If there are any output models, they will be saved in that path
                    # and the output_model.config["model_path"] would like /olive-ws/runner_output/model.onnx
                    # the model path should starts with /olive-ws/runner_output
                    assert resource_str.startswith(docker_output_path)
                    candidate_resource_path = resource_str.replace(docker_output_path, output_model_path)
                    output_model.config[resource_name] = candidate_resource_path

                logger.debug("Model path is: %s", output_model.config["model_path"])
                return output_model
        else:
            logger.error("Model output file %s not found.", model_output_json_file)
            return None

    def evaluate_model(
        self, model_config: "ModelConfig", evaluator_config: "OliveEvaluatorConfig", accelerator: "AcceleratorSpec"
    ) -> Dict[str, Any]:
        container_root_path = Path("/olive-ws/")
        with tempfile.TemporaryDirectory() as tempdir:
            metric_json = self._run_eval_container(
                tempdir, model_config, evaluator_config, accelerator, container_root_path
            )
            if metric_json.is_file():
                with metric_json.open() as f:
                    metrics_res = json.load(f)
                    return MetricResult.parse_obj(metrics_res)
            else:
                logger.error("Metric result file %s not found.", metric_json)
                return None

    def _run_eval_container(
        self,
        workdir,
        model_config: "ModelConfig",
        evaluator_config: "OliveEvaluatorConfig",
        accelerator: "AcceleratorSpec",
        container_root_path: Path,
    ):
        eval_output_path = "eval_output"
        eval_output_name = "eval_res.json"

        volumes_list = []
        # mount eval script
        eval_file_mount_path, eval_file_mount_str = docker_utils.create_eval_script_mount(container_root_path)
        volumes_list.append(eval_file_mount_str)

        # mount dev stuff
        if self.is_dev:
            _, dev_mount_str = docker_utils.create_dev_mount(workdir, container_root_path)
            volumes_list.append(dev_mount_str)

        # mount model
        model_mounts, model_mount_str_list, _ = docker_utils.create_model_mount(
            model_config=model_config, container_root_path=container_root_path
        )
        volumes_list += model_mount_str_list

        metrics_copy = copy.deepcopy(evaluator_config.metrics)
        # mount metrics related external files
        volumes_list.extend(
            # the metrics_copy is modified when creating the volumes list
            docker_utils.create_metric_volumes_list(
                metrics=metrics_copy,
                container_root_path=container_root_path,
            )
        )

        # mount config file
        data = self._create_eval_config(model_config, metrics_copy, model_mounts)
        config_mount_path, config_file_mount_str = docker_utils.create_config_file(
            workdir=workdir,
            config=data,
            container_root_path=container_root_path,
        )
        volumes_list.append(config_file_mount_str)

        output_local_path, output_mount_path, output_mount_str = docker_utils.create_output_mount(
            workdir=workdir,
            docker_output_path=eval_output_path,
            container_root_path=container_root_path,
        )
        volumes_list.append(output_mount_str)
        logger.debug("The volumes list is %s", volumes_list)

        eval_command = docker_utils.create_evaluate_command(
            eval_script_path=eval_file_mount_path,
            config_path=config_mount_path,
            output_path=output_mount_path,
            output_name=eval_output_name,
            accelerator=accelerator,
        )
        return self._run_container(eval_command, volumes_list, output_local_path, eval_output_name, accelerator)

    @staticmethod
    def _create_eval_config(model_config: "ModelConfig", metrics: List["Metric"], model_mounts: Dict[str, str]):
        model_json = model_config.to_json(check_object=True)
        for k, v in model_mounts.items():
            model_json["config"][k] = v

        return {"metrics": [k.to_json(check_object=True) for k in metrics], "model": model_json}

    @staticmethod
    def _create_runner_config(
        model_config: "ModelConfig",
        pass_config: Dict[str, Any],
        model_mounts: Dict[str, str],
        data_mounts: Dict[str, str],
    ):
        model_json = model_config.to_json(check_object=True)
        for k, v in model_mounts.items():
            model_json["config"][k] = v

        pass_config_copy = copy.deepcopy(pass_config)
        for k, v in data_mounts.items():
            pass_config_copy["config"][k] = v

        return {"model": model_json, "pass": pass_config_copy}

    def _run_container(
        self,
        command,
        volumes_list: List[str],
        output_local_path,
        output_name,
        accelerator: "AcceleratorSpec",
    ):
        run_command = docker_utils.create_run_command(run_params=self.run_params)

        environment = run_command.pop("environment", {})
        envs_dict = {"PYTHONPYCACHEPREFIX": "/tmp"}
        for k, v in envs_dict.items():
            if isinstance(environment, list):
                environment = {env.split("=")[0]: env.split("=")[1] for env in environment}
            elif isinstance(environment, dict) and not environment.get(k):
                environment[k] = v
        if self.hf_token:
            token = get_huggingface_token()
            environment.update({"HF_TOKEN": token})

        log_level = logging.getLevelName(logger.getEffectiveLevel())
        environment.update({"OLIVE_LOG_LEVEL": log_level})

        logger.debug("Running container with command: %s", command)
        if accelerator.accelerator_type == Device.GPU:
            run_command["device_requests"] = [docker.types.DeviceRequest(capabilities=[["gpu"]])]

        container = self.docker_client.containers.run(
            image=self.image,
            command=command,
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
            raise ContainerError(
                container, exit_code, command, self.image, f"Docker container evaluation failed with: {error_msg}"
            )
        logger.debug("Docker container run completed successfully")

        return Path(output_local_path) / output_name

    def _create_data_mounts_for_pass(self, container_root_path: Path, the_pass: "Pass"):
        mounts = {}
        mount_strs = []
        config_dict = the_pass.config.dict()
        for param, _, category in the_pass.path_params:
            param_val = config_dict.get(param)
            if category == ParamCategory.DATA and param_val:
                mount = str(container_root_path / param)
                mounts[param] = mount
                mount_strs.append(f"{param_val}:{mount}")

        return mounts, mount_strs

    def remove(self):
        self.docker_client.images.remove(self.image.tags[0], force=True)
        logger.info("Image %s removed successfully.", self.image.tags[0])


def _print_docker_logs(logs, level=logging.DEBUG):
    msgs = []
    for log in logs:
        if "stream" in log:
            msgs.append(str(log["stream"]).strip())
        else:
            msgs.append(str(log).strip())

    message = "\n".join(msgs)
    logger.log(level, message)


def get_huggingface_token():
    """Get huggingface token from environment variable or token file."""
    import os

    if os.getenv("HF_TOKEN"):
        return os.getenv("HF_TOKEN")

    token_path = Path.home() / ".huggingface" / "token"
    if not token_path.exists():
        logger.error(
            "Huggingface token is required at this step. Could not find huggingface token at %s. "
            "Please login to huggingface first using `huggingface-cli login`. "
            "If you already logged in, Olive will get token from '~/.huggingface/token' file'. "
            "Please make sure the token file exists.",
            token_path,
        )
        return None
    with Path(token_path).open() as f:
        return f.read().strip()
