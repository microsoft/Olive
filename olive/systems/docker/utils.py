# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import shutil
from pathlib import Path
from typing import List

from olive.cache import get_local_path_from_root
from olive.evaluator.metric import Metric
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig

logger = logging.getLogger(__name__)


def create_config_file(
    tempdir, model_config: ModelConfig, metrics: List[Metric], container_root_path: Path, model_mounts: dict
):
    model_json = model_config.to_json(check_object=True)
    for k, v in model_mounts.items():
        model_json["config"][k] = v

    config_file_path = Path(tempdir) / "config.json"
    data = {"metrics": [k.dict() for k in metrics], "model": model_json}

    # the config yaml file saved to local disk
    with config_file_path.open("w") as f:
        json.dump(data, f)
    config_mount_path = str(container_root_path / "config.json")
    config_file_mount_str = f"{config_file_path}:{config_mount_path}"
    return config_mount_path, config_file_mount_str


def create_evaluate_command(
    eval_script_path: str, config_path: str, output_path: str, output_name: str, accelerator: AcceleratorSpec
):
    # no need to pass model_path since it's already updated in config file
    parameters = [
        f"--config {config_path}",
        f"--output_path {output_path}",
        f"--output_name {output_name}",
        f"--accelerator_type {accelerator.accelerator_type}",
        f"--execution_provider {accelerator.execution_provider}",
    ]
    return f"python {eval_script_path} {' '.join(parameters)}"


def create_run_command(run_params: dict):
    if not run_params:
        return {}
    run_command_dict = {}
    for k, v in run_params.items():
        run_command_dict[k.replace("-", "_")] = v
    return run_command_dict


def create_metric_volumes_list(
    data_root: str, metrics: List[Metric], container_root_path: Path, mount_list: list
) -> List[str]:
    for metric in metrics:
        metric_path = container_root_path / "metrics" / metric.name
        if metric.user_config.user_script:
            user_script_path = str(Path(metric.user_config.user_script).resolve())
            user_script_name = Path(metric.user_config.user_script).name
            user_script_mount_path = str(metric_path / user_script_name)

            mount_list.append(f"{user_script_path}:{user_script_mount_path}")
            metric.user_config.user_script = user_script_mount_path

        if metric.user_config.script_dir:
            script_dir_path = str(Path(metric.user_config.script_dir).resolve())
            script_dir_name = Path(metric.user_config.script_dir).name
            script_dir_mount_path = str(metric_path / script_dir_name)

            mount_list.append(f"{script_dir_path}:{script_dir_mount_path}")
            metric.user_config.script_dir = script_dir_mount_path

        if data_root or metric.user_config.data_dir:
            data_dir = get_local_path_from_root(data_root, metric.user_config.data_dir)
            mount_list.append(f"{data_dir}:{str(metric_path / 'data_dir')}")
            metric.user_config.data_dir = str(metric_path / "data_dir")

    return mount_list


def create_model_mount(model_config: ModelConfig, container_root_path: Path):
    mounts = {}
    mount_strs = []
    resource_paths = model_config.get_resource_paths()
    for resource_name, resource_path in resource_paths.items():
        # if the resource path is None or string name, we need not to mount it
        if not resource_path or resource_path.is_string_name():
            continue

        relevant_path = resource_path.get_path()
        resource_path_mount_path = str(container_root_path / Path(relevant_path).name)
        resource_path_mount_str = f"{str(Path(relevant_path).resolve())}:{resource_path_mount_path}"
        mounts[resource_name] = resource_path_mount_path
        mount_strs.append(resource_path_mount_str)
    return mounts, mount_strs


def create_eval_script_mount(container_root_path: Path):
    eval_file_mount_path = str(container_root_path / "eval.py")
    current_dir = Path(__file__).resolve().parent
    eval_file_mount_str = f"{str(current_dir / 'eval.py')}:{eval_file_mount_path}"
    return eval_file_mount_path, eval_file_mount_str


def create_dev_mount(tempdir: Path, container_root_path: Path):
    logger.warning(
        "Dev mode is only enabled for CI pipeline! "
        "It will overwrite the Olive package in docker container with latest code."
    )
    tempdir = Path(tempdir)

    # copy the whole project folder to tempdir
    project_folder = Path(__file__).resolve().parent.parent.parent
    shutil.copytree(project_folder, tempdir / "olive", ignore=shutil.ignore_patterns("__pycache__"))

    project_folder_mount_path = str(container_root_path / "olive")
    project_folder_mount_str = f"{tempdir / 'olive'}:{project_folder_mount_path}"
    return project_folder_mount_path, project_folder_mount_str


def create_output_mount(tempdir, docker_eval_output_path: str, container_root_path: Path):
    output_local_path = Path(tempdir) / docker_eval_output_path
    output_local_path.mkdir(parents=True, exist_ok=True)
    output_mount_path = str(container_root_path / docker_eval_output_path)
    output_mount_str = f"{output_local_path}:{output_mount_path}"
    return output_local_path, output_mount_path, output_mount_str
