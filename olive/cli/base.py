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
from olive.common.user_module_loader import UserModuleLoader
from olive.common.utils import hash_dict


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


def _get_hf_input_model(args, model_path):
    print("Loading HuggingFace model from:", model_path)
    input_model = {
        "type": "HfModel",
        "model_path": model_path,
        "load_kwargs": {
            "trust_remote_code": args.trust_remote_code,
            "attn_implementation": "eager",
        },
    }
    if args.task:
        input_model["task"] = args.task
    return input_model


def _get_onnx_input_model(model_path):
    print("Loading ONNX model from:", model_path)
    return {
        "type": "OnnxModel",
        "model_path": model_path,
    }


def _get_pt_input_model(args, model_path):
    if not args.model_script:
        raise ValueError("model_script is not provided. Either model_name_or_path or model_script is required.")

    print("Loading PyTorch model from:", model_path)
    user_module_loader = UserModuleLoader(args.model_script, args.script_dir)
    input_model_config = {
        "type": "PyTorchModel",
        "model_script": args.model_script,
    }

    if args.script_dir:
        input_model_config["script_dir"] = args.script_dir

    # model path has high priority than model loader
    if model_path:
        input_model_config["model_path"] = model_path
    else:
        if not user_module_loader.has_function("_model_loader"):
            raise ValueError(
                "_model_loader is required for PyTorch model in the script if model_name_or_path is not provided."
            )
        input_model_config["model_loader"] = "_model_loader"

    model_funcs = [
        ("io_config", "_io_config"),
        ("dummy_inputs_func", "_dummy_inputs"),
        ("model_file_format", "_model_file_format"),
    ]
    input_model_config.update(
        {config_key: func_name for config_key, func_name in model_funcs if user_module_loader.has_function(func_name)}
    )

    if "io_config" not in input_model_config and "dummy_inputs_func" not in input_model_config:
        raise ValueError("_io_config or _dummy_inputs is required in the script for PyTorch model.")
    return input_model_config


def get_input_model_config(args) -> Union[str, Dict[str, str]]:
    """Parse the model_name_or_path and return the model name or path.

    Check model_name_or_path formats in order:
    1. Local PyTorch model with model loader
    2. azureml://registries/<registry_name>/models/<model_name>/versions/<version> (only for HF model)
    3. https://huggingface.co/<model_name> (only for HF model)
    4. azureml:<model_name>:<version> (only for PyTorch model)
    5. HF model name string
    6. local file path
      a. local onnx model file path (either a user-provided model or a model produced by the Olive CLI)
      b. local PyTorch model file path (either a user-provided model or a model produced by the Olive CLI)
      c. local HF model file path (either a user-provided model or a model produced by the Olive CLI)
    """
    model_name_or_path = args.model_name_or_path

    # Check if local PyTorch model with model loader
    if model_name_or_path is None:
        print("model_name_or_path is not provided. Using model_script to load the model.")
        return _get_pt_input_model(args, None)

    # Check AzureML Registry model
    pattern = (
        r"^azureml://registries/(?P<registry_name>[^/]+)/models/(?P<model_name>[^/]+)/versions/(?P<version>[^/]+)$"
    )
    match = re.match(pattern, model_name_or_path)
    if match:
        return _get_hf_input_model(
            args,
            {
                "type": "azureml_registry_model",
                "registry_name": match.group("registry_name"),
                "name": match.group("model_name"),
                "version": match.group("version"),
            },
        )

    # Check HuggingFace url
    pattern = r"https://huggingface\.co/([^/]+/[^/]+)(?:/.*)?"
    match = re.search(pattern, model_name_or_path)
    if match:
        return _get_hf_input_model(args, match.group(1))

    # Check AzureML model
    pattern = r"^azureml:(?P<model_name>[^:]+):(?P<version>[^:]+)$"
    match = re.match(pattern, model_name_or_path)
    if match:
        return _get_pt_input_model(
            args,
            {
                "type": "azureml_model",
                "name": match.group("model_name"),
                "version": match.group("version"),
            },
        )

    model_path = Path(model_name_or_path)

    # Check HF model name string
    if not model_path.exists():
        return _get_hf_input_model(args, model_name_or_path)

    # Check if local model is from Olive CLI
    if model_path.is_dir():
        for file in model_path.iterdir():
            if file.is_file() and file.name == "model_config.json":
                with open(file) as f:
                    return json.load(f)

    # Check local pt file (user-provided model)
    if model_path.is_file():
        if model_path.suffix in (".pt", ".pth"):
            return _get_pt_input_model(args, model_name_or_path)
        if model_path.suffix == ".onnx":
            return _get_onnx_input_model(model_name_or_path)

    # Check local HF model file (user-provided model)
    return _get_hf_input_model(args, model_name_or_path)


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


def add_model_options(sub_parser):
    model_group = sub_parser.add_argument_group("Model options")
    model_group.add_argument(
        "-m",
        "--model_name_or_path",
        type=str,
        help=(
            "The model checkpoint for weights initialization. If using an AzureML Registry model, provide the model"
            " path as 'registry_name:model_name:version'."
        ),
    )
    model_group.add_argument("--trust_remote_code", action="store_true", help="Trust remote code when loading a model.")
    model_group.add_argument("-t", "--task", type=str, help="Task for which the model is used.")
    model_group.add_argument(
        "--model_script",
        type=str,
        help="The script file containing the model definition. Required for PyTorch model.",
    )
    model_group.add_argument(
        "--script_dir",
        type=str,
        default=None,
        help="The directory containing the model script file.",
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


# TODO(team): Remove this function once the output structure is refactored
def get_output_model_number(outputs: Dict) -> int:
    return sum(len(f.nodes) for f in outputs.values())


# TODO(team): Remove this function once the output structure is refactored
def save_model_config(outputs: Dict, output_path: Path):
    pf_footprint = next(iter(outputs.values()))
    model_config = next(iter(pf_footprint.get_top_ranked_nodes(1))).model_config
    model_path = model_config["config"]["model_path"]
    model_config["config"]["model_path"] = str(output_path.resolve() / Path(model_path).name)
    model_config_path = output_path / "model_config.json"
    with open(model_config_path, "w") as f:
        json.dump(model_config, f, indent=4)
