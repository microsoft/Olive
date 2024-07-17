# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, List

from olive.common.utils import copy_dir

if TYPE_CHECKING:
    from olive.evaluator.metric import Metric
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ModelConfig

logger = logging.getLogger(__name__)


def create_config_file(workdir, config: dict, container_root_path: Path):
    config_file_path = Path(workdir) / "config.json"
    # the config yaml file saved to local disk
    with config_file_path.open("w") as f:
        json.dump(config, f, indent=4)
    config_mount_path = str(container_root_path / "config.json")
    config_file_mount_str = f"{config_file_path}:{config_mount_path}"
    return config_mount_path, config_file_mount_str


def create_runner_command(runner_script_path: str, config_path: str, output_path: str, output_name: str):
    command = [
        "python",
        runner_script_path,
        "--config",
        config_path,
        "--output_path",
        output_path,
        "--output_name",
        output_name,
    ]
    return " ".join(command)


def create_evaluate_command(
    eval_script_path: str, config_path: str, output_path: str, output_name: str, accelerator: "AcceleratorSpec"
):
    # no need to pass model_path since it's already updated in config file
    parameters = [
        f"--config {config_path}",
        f"--output_path {output_path}",
        f"--output_name {output_name}",
        f"--accelerator_type {accelerator.accelerator_type}",
    ]
    if accelerator.execution_provider:
        parameters.append(f"--execution_provider {accelerator.execution_provider}")

    return f"python {eval_script_path} {' '.join(parameters)}"


def create_run_command(run_params: dict):
    if not run_params:
        return {}
    run_command_dict = {}
    for k, v in run_params.items():
        run_command_dict[k.replace("-", "_")] = v
    return run_command_dict


def create_metric_volumes_list(metrics: List["Metric"], container_root_path: Path) -> List[str]:
    volume_list = []
    for metric in metrics:
        metric_path = container_root_path / "metrics" / metric.name
        if metric.user_config.user_script:
            user_script_path = str(Path(metric.user_config.user_script).resolve())
            user_script_name = Path(metric.user_config.user_script).name
            user_script_mount_path = str(metric_path / user_script_name)

            volume_list.append(f"{user_script_path}:{user_script_mount_path}")
            metric.user_config.user_script = user_script_mount_path

        if metric.user_config.script_dir:
            script_dir_path = str(Path(metric.user_config.script_dir).resolve())
            script_dir_name = Path(metric.user_config.script_dir).name
            script_dir_mount_path = str(metric_path / script_dir_name)

            volume_list.append(f"{script_dir_path}:{script_dir_mount_path}")
            metric.user_config.script_dir = script_dir_mount_path

        if metric.data_config:
            if metric.data_config.load_dataset_params.get("data_dir"):
                data_dir_path = str(Path(metric.data_config.load_dataset_params.get("data_dir")).resolve())
                volume_list.append(f"{data_dir_path}:{str(metric_path / 'data_dir')}")
                metric.data_config.load_dataset_params["data_dir"] = str(metric_path / "data_dir")

            if metric.data_config.user_script:
                user_script_path = str(Path(metric.data_config.user_script).resolve())
                user_script_name = Path(metric.data_config.user_script).name
                user_script_mount_path = str(metric_path / user_script_name)

                volume_list.append(f"{user_script_path}:{user_script_mount_path}")
                metric.data_config.user_script = user_script_mount_path

            if metric.data_config.script_dir:
                script_dir_path = str(Path(metric.data_config.script_dir).resolve())
                script_dir_name = Path(metric.data_config.script_dir).name
                script_dir_mount_path = str(metric_path / script_dir_name)

                volume_list.append(f"{script_dir_path}:{script_dir_mount_path}")
                metric.data_config.script_dir = script_dir_mount_path

    return volume_list


def create_model_mount(model_config: "ModelConfig", container_root_path: Path):
    mounts = {}
    mount_strs = []
    mount_to_local = {}
    resource_paths = model_config.get_resource_paths()
    for resource_name, resource_path in resource_paths.items():
        # if the resource path is None or string name, we need not to mount it
        if not resource_path or resource_path.is_string_name():
            continue

        relevant_path = resource_path.get_path()
        local_path = str(Path(relevant_path).resolve())
        resource_path_mount_path = str(container_root_path / Path(relevant_path).name)
        resource_path_mount_str = f"{local_path}:{resource_path_mount_path}"
        mounts[resource_name] = resource_path_mount_path
        mount_strs.append(resource_path_mount_str)
        mount_to_local[resource_path_mount_path] = local_path
    return mounts, mount_strs, mount_to_local


def create_runner_script_mount(container_root_path: Path):
    runner_file_mount_path = str(container_root_path / "runner.py")
    current_dir = Path(__file__).resolve().parent
    runner_file_mount_str = f"{str(current_dir / 'runner.py')}:{runner_file_mount_path}"
    return runner_file_mount_path, runner_file_mount_str


def create_eval_script_mount(container_root_path: Path):
    eval_file_mount_path = str(container_root_path / "eval.py")
    current_dir = Path(__file__).resolve().parent
    eval_file_mount_str = f"{str(current_dir / 'eval.py')}:{eval_file_mount_path}"
    return eval_file_mount_path, eval_file_mount_str


def create_dev_mount(workdir: Path, container_root_path: Path):
    logger.warning(
        "Dev mode is only enabled for CI pipeline! "
        "It will overwrite the Olive package in docker container with latest code."
    )
    workdir = Path(workdir)

    # copy the whole project folder to tempdir
    project_folder = Path(__file__).resolve().parents[2]
    copy_dir(project_folder, workdir / "olive", ignore=shutil.ignore_patterns("__pycache__"))

    project_folder_mount_path = str(container_root_path / "olive")
    project_folder_mount_str = f"{workdir / 'olive'}:{project_folder_mount_path}"
    return project_folder_mount_path, project_folder_mount_str


def create_output_mount(workdir, docker_output_path: str, container_root_path: Path):
    output_local_path = Path(workdir) / docker_output_path
    output_local_path.mkdir(parents=True, exist_ok=True)
    output_mount_path = str(container_root_path / docker_output_path)
    output_mount_str = f"{output_local_path}:{output_mount_path}"
    return output_local_path, output_mount_path, output_mount_str
