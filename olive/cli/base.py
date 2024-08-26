# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import logging
import subprocess
from abc import ABC, abstractmethod
from argparse import ArgumentParser, Namespace
from pathlib import Path
from typing import ClassVar, Optional

import yaml

from olive.cli.constants import CONDA_CONFIG
from olive.common.utils import hash_dict

logger = logging.getLogger(__name__)


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


def is_remote_run(args):
    return all([args.resource_group, args.workspace_name, args.aml_compute])


def update_remote_option(config, args, cli_action, tempdir):
    if args.resource_group or args.workspace_name or args.aml_compute:
        if not is_remote_run(args):
            raise ValueError("resource_group, workspace_name and aml_compute are required for remote workflow run.")

        config["workflow_id"] = f"{cli_action}-{hash_dict(config)}"

        try:
            subscription_id = json.loads(subprocess.check_output("az account show", shell=True).decode("utf-8"))["id"]
            logger.info("Using Azure subscription ID: %s", subscription_id)

        except subprocess.CalledProcessError:
            logger.exception(
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
