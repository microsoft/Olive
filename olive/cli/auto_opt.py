# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Dict

from olive.cli.base import (
    BaseOliveCLICommand,
)
from olive.common.utils import IntEnumBase, set_nested_dict_value, set_tempdir

logger = logging.getLogger(__name__)

class AutoOptCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "auto-opt",
            help=("Automatically performance optimize input model"),
        )

        # model options
        model_group = sub_parser.add_argument_group("model options")
        model_group.add_argument(
            "--model",
            required=True,
            help="Onnx input model path.",
        )
        sub_parser.add_argument(
            "--provider", 
            type=str, 
            default="CUDAExecutionProvider",
            choices=["CPU", "CUDAExecutionProvider", "DML","VitisAI"],
            help="OnnxRuntime Execution Provider to use"
        )
        sub_parser.add_argument(
            "--precision",
            type=str,
            default="fp16",
            choices=["fp16", "fp32", "int4", "int8"],
            help="The precision of the ONNX model. This is used by Model Builder"
        )
        sub_parser.add_argument("-o", "--output_path", type=str, default="onnx-model", help="Output path")
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )

        sub_parser.set_defaults(func=AutoOptCommand)

    def run(self):
        from olive.workflows import run as olive_run

        set_tempdir(self.args.output_path)

        with tempfile.TemporaryDirectory() as tempdir:
            run_config = self.get_run_config(tempdir)

            olive_run(run_config)

            if is_remote_run(self.args):
                # TODO(jambayk): point user to datastore with outputs or download outputs
                # both are not implemented yet
                return

            output_path = Path(self.args.output_path)
            logger.info("Optimized ONNX Model is saved to %s", output_path.resolve())

    def get_run_config(self, tempdir: str) -> Dict:
        config = deepcopy(TEMPLATE)

        config["input_model"]["model_path"] = self.args.model
        config["output_dir"] = self.args.output_path
        ep = {}
        ep["execution_providers"] = [self.args.provider]
        #config["systems"]["local_system"]["accelerators"]= [ep]
        d = {}
        d["device"] = "cpu"
        config["systems"]["local_system"]["accelerators"]= [d,ep]
        config["auto_optimizer_config"]["precision"] = self.args.precision

        print(f'........ {config}')
        return config


TEMPLATE = {
    "input_model": {"type" : "ONNXModel"},
    "auto_optimizer_config": {},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"execution_providers": ["CUDAExecutionProvider"]}],
        }
    },
    "host": "local_system",
    "target": "local_system",
}
