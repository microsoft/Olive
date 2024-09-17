# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

import json
import re
import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import ClassVar, Dict, Optional, Union

import yaml

from olive.cli.constants import CONDA_CONFIG
from olive.common.utils import hardlink_copy_dir, hash_dict, set_nested_dict_value
from olive.resource_path import find_all_resources


class BaseOliveCLICommand(ABC):
    allow_unknown_args: ClassVar[bool] = False

    def __init__(self, parser: ArgumentParser, args: Namespace, unknown_args: Optional[list] = None):
        self.args = args
        self.unknown_args = unknown_args

        if unknown_args and not self.allow_unknown_args:
            parser.error(f"Unknown arguments: {unknown_args}")

    @staticmethod
    @abstractmethod
    def register_subcommand(parser: ArgumentParser):
        raise NotImplementedError

    @abstractmethod
    def run(self):
        raise NotImplementedError


def get_model_name_or_path(model_name_or_path) -> Union[str, Dict[str, str]]:
    pattern = r"^(?P<registry_name>[^:]+):(?P<model_name>[^:]+):(?P<version>[^:]+)$"
    match = re.match(pattern, model_name_or_path)

    if match:
        return {
            "type": "azureml_registry_model",
            "registry_name": match.group("registry_name"),
            "name": match.group("model_name"),
            "version": match.group("version"),
        }

    pattern = r"https://huggingface\.co/([^/]+/[^/]+)(?:/.*)?"
    match = re.search(pattern, model_name_or_path)

    if match:
        return match.group(1)

    return model_name_or_path


def add_logging_options(sub_parser):
    log_group = sub_parser.add_argument_group("logging options")
    log_group.add_argument(
        "--log_level",
        type=int,
        default=3,
        help="Logging level. Default is 3. level 0: DEBUG, 1: INFO, 2: WARNING, 3: ERROR, 4: CRITICAL",
    )


def add_remote_options(sub_parser):
    remote_group = sub_parser.add_argument_group("remote options")
    remote_group.add_argument(
        "--resource_group",
        type=str,
        required=False,
        help="Resource group for the AzureML workspace.",
    )
    remote_group.add_argument(
        "--workspace_name",
        type=str,
        required=False,
        help="Workspace name for the AzureML workspace.",
    )
    remote_group.add_argument(
        "--keyvault_name",
        type=str,
        required=False,
        help=(
            "The azureml keyvault name with huggingface token to use for remote run. Refer to"
            " https://microsoft.github.io/Olive/features/huggingface_model_optimization.html#huggingface-login for"
            " more details."
        ),
    )
    remote_group.add_argument(
        "--aml_compute",
        type=str,
        required=False,
        help="The compute name to run the workflow on.",
    )


def add_hf_model_options(sub_parser):
    model_group = sub_parser.add_argument_group("model options")
    model_group.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        required=True,
        help=(
            "The model checkpoint for weights initialization. If using an AzureML Registry model, provide the model"
            " path as 'registry_name:model_name:version'."
        ),
    )
    model_group.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading a model.")
    model_group.add_argument("-t", "--task", type=str, help="Task for which the model is used.")


def is_remote_run(args):
    return all([args.resource_group, args.workspace_name, args.aml_compute])


def update_remote_option(config, args, cli_action, tempdir):
    if args.resource_group or args.workspace_name or args.aml_compute:
        if not is_remote_run(args):
            raise ValueError("resource_group, workspace_name and aml_compute are required for remote workflow run.")

        config["workflow_id"] = f"{cli_action}-{hash_dict(config)}"

        try:
            subscription_id = json.loads(subprocess.check_output("az account show", shell=True).decode("utf-8"))["id"]
            print("Using Azure subscription ID: %s", subscription_id)

        except subprocess.CalledProcessError:
            print(
                "Error: Unable to retrieve account information. "
                "Make sure you are logged in to Azure CLI with command `az login`."
            )

        config["azureml_client"] = {
            "subscription_id": subscription_id,
            "resource_group": args.resource_group,
            "workspace_name": args.workspace_name,
            "keyvault_name": args.keyvault_name,
            "default_auth_params": {"exclude_managed_identity_credential": True},
        }

        conda_file_path = Path(tempdir) / "conda_gpu.yaml"
        with open(conda_file_path, "w") as f:
            yaml.dump(CONDA_CONFIG, f)

        config["systems"]["aml_system"] = {
            "type": "AzureML",
            "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
            "aml_compute": args.aml_compute,
            "aml_docker_config": {
                "base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04",
                "conda_file_path": str(conda_file_path),
            },
            "hf_token": bool(args.keyvault_name),
        }
        config["workflow_host"] = "aml_system"


def save_output_model(config: Dict, output_model_dir: Union[str, Path]):
    run_output_path = Path(config["output_dir"]) / "output_model"
    if not run_output_path.exists():
        print("Command failed. Please set the log_level to 1 for more detailed logs.")
        return

    output_model_dir = Path(output_model_dir).resolve()

    hardlink_copy_dir(run_output_path, output_model_dir)

    # need to update the local path in the model_config.json
    # should the path be relative or absolute? relative makes it easy to move the output
    # around but the path needs to be updated when the model config is used
    model_config_path = output_model_dir / "model_config.json"
    with model_config_path.open("r") as f:
        model_config = json.load(f)

    all_resources = find_all_resources(model_config)
    for resource_key, resource_path in all_resources.items():
        resource_path_str = resource_path.get_path()
        if resource_path_str.startswith(str(run_output_path)):
            set_nested_dict_value(
                model_config,
                resource_key,
                resource_path_str.replace(str(run_output_path), str(output_model_dir)),
            )

    with model_config_path.open("w") as f:
        json.dump(model_config, f, indent=4)

    print(f"Command succeeded. Output model saved to {output_model_dir}")


# TODO(team): Remove this function once the output structure is refactored
def get_output_model_number(outputs: Dict) -> int:
    return sum(len(f.nodes) for f in outputs.values())
