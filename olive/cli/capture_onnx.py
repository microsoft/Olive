# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import codecs
import json
import logging
import re
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Dict, Union

import yaml

from olive.cli.base import BaseOliveCLICommand
from olive.common.utils import hardlink_copy_dir, hash_dict, set_nested_dict_value, set_tempdir

logger = logging.getLogger(__name__)


class CaptureOnnxGraphCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "capture-onnx-graph",
            help=(
                "Capture ONNX graph using PyTorch Exporter or Model Builder from the Huggingface model."
            ),
        )

        sub_parser.add_argument(
            "--use_dynamo_exporter",
            type=bool,
            default=False,
            help="Whether to use dynamo_export API to export ONNX model.",
        )

        sub_parser.add_argument(
            "--use_model_builder",
            action="store_true",
            help="Whether to use Model Builder to capture ONNX model.",
        )

        sub_parser.add_argument(
            "--past_key_value_name",
            type=str,
            default="past_key_values",
            help=(
                "The arguments name to point to past key values. For model loaded from huggingface, "
                "it is 'past_key_values'. Basically, it is used only when `use_dynamo_exporter` is True."
            )
        )

        sub_parser.add_argument(
            "--device",
            type=str,
            help=(
                "The device to use to trace the model, e.g., 'cuda' or 'cpu'. If not specified, will"
                " use 'cpu' for PyTorch model and 'cuda' for the distributed model."
            ),
        )

        sub_parser.add_argument(
            "--torch_dtype",
            type=str,
            help=(
                "The dtype to cast the model to before capturing the ONNX graph, e.g., 'float32' or 'float16'."
                " If not specified will use the model as is."
            ),
        )

        sub_parser.add_argument(
            "--use_ort_genai",
            type=bool,
            default=False,
            help="Use OnnxRuntie generate() API to run the model"
        )

        # directory options
        sub_parser.add_argument("-o", "--output_path", type=str, default="onnx-model", help="Output path")
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )

        # model options
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

        model_group.add_argument(
            "-t",
            "--task",
            type=str,
            help="Task for which the model is used."
        )
        sub_parser.set_defaults(func=CaptureOnnxGraphCommand)

    def run(self):
        from olive.workflows import run as olive_run

        set_tempdir(self.args.tempdir)

        with tempfile.TemporaryDirectory() as tempdir:
            run_config = self.get_run_config(tempdir)

            olive_run(run_config)

    def get_model_name_or_path(self) -> Union[str, Dict]:
        pattern = r"(?P<registry_name>[^:]+):(?P<model_name>[^:]+):(?P<version>[^:]+)"
        match = re.match(pattern, self.args.model_name_or_path)
        if not match:
            return self.args.model_name_or_path

        return {
            "type": "azureml_registry_model",
            "registry_name": match.group("registry_name"),
            "name": match.group("model_name"),
            "version": match.group("version"),
        }

    def get_run_config(self, tempdir: str) -> Dict:
        config = deepcopy(TEMPLATE)

        config["input_model"]["model_path"] = self.get_model_name_or_path()
        if self.args.task is not None:
            config["input_model"]["task"] = self.args.task

        if self.args.use_model_builder:
            del config["passes"]["c"]
        else:
            del config["passes"]["m"]
            config["passes"]["c"]["save_metadata_for_token_generation"] = self.args.use_ort_genai

        print(f'.... {config}')
        return config


TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu"}],
        }
    },
    "passes": {
        "c": {
            "type": "OnnxConversion",
            "target_opset": 17,
            "torch_dtype": "float32",
            "save_metadata_for_token_generation": False,
        },
        "m": {"type": "ModelBuilder", "metadata_only": False},
    },
    "host": "local_system",
    "target": "local_system",
}