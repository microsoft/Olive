# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
from pathlib import Path
from typing import List

from olive.constants import Framework
from olive.evaluator.metric import Metric, MetricList
from olive.model import OliveModel

logger = logging.getLogger(__name__)


def create_config_file(tempdir, model: OliveModel, metrics: List[Metric], container_root_path: Path):
    model_json = model.to_json(check_object=True)

    config_file_path = Path(tempdir) / "config.json"
    metric_json_list = MetricList(__root__=metrics).json()
    data = {"metrics": metric_json_list, "model": model_json}

    # the config yaml file saved to local disk
    with config_file_path.open("w") as f:
        json.dump(data, f)
    config_mount_path = str(container_root_path / "config.json")
    config_file_mount_str = f"{config_file_path}:{config_mount_path}"
    return config_mount_path, config_file_mount_str


def create_evaluate_command(
    eval_script_path: str, model_path: str, config_path: str, output_path: str, output_name: str
):
    parameters = [
        f"--config {config_path}",
        f"--model_path {model_path}",
        f"--output_path {output_path}",
        f"--output_name {output_name}",
    ]
    cmd_line = f"python {eval_script_path} {' '.join(parameters)}"
    return cmd_line


def create_run_command(run_params: dict):
    if not run_params:
        return {}
    run_command_dict = {}
    for k, v in run_params.items():
        run_command_dict[k.replace("-", "_")] = v
    return run_command_dict


def create_metric_volumes_list(metrics: List[Metric], container_root_path: Path, mount_list: list) -> List[str]:
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

        if metric.user_config.data_dir:
            data_dir = str(Path(metric.user_config.data_dir).resolve())
            mount_list.append(f"{data_dir}:{str(metric_path / 'data_dir')}")
            metric.user_config.data_dir = str(metric_path / "data_dir")

    return mount_list


def create_model_mount(model: OliveModel, container_root_path: Path):
    model_mount_path = str(container_root_path / Path(model.model_path).name)
    model_mount_str = f"{str(Path(model.model_path).resolve())}:{model_mount_path}"
    model.model_path = model_mount_path
    model_mount_str_list = [model_mount_str]

    if model.framework == Framework.PYTORCH:
        if model.script_dir:
            script_dir_mount = f"{model.script_dir}:{container_root_path}"
            model.script_dir = container_root_path
            model_mount_str_list.append(script_dir_mount)
    return model_mount_path, model_mount_str_list


def create_eval_script_mount(container_root_path: Path):
    eval_file_mount_path = str(container_root_path / "eval.py")
    current_dir = Path(__file__).resolve().parent
    eval_file_mount_str = f"{str(current_dir / 'eval.py')}:{eval_file_mount_path}"
    return eval_file_mount_path, eval_file_mount_str


def create_output_mount(tempdir, docker_eval_output_path: str, container_root_path: Path):
    output_local_path = Path(tempdir) / docker_eval_output_path
    output_local_path.mkdir(parents=True, exist_ok=True)
    output_mount_path = str(container_root_path / docker_eval_output_path)
    output_mount_str = f"{output_local_path}:{output_mount_path}"
    return output_local_path, output_mount_path, output_mount_str
