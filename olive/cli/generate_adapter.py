# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_save_config_file_options,
    add_shared_cache_options,
    get_input_model_config,
    update_remote_options,
    update_shared_cache_options,
)
from olive.common.utils import WeightsFileFormat, set_nested_dict_value


class GenerateAdapterCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "generate-adapter", help="Generate ONNX model with adapters as inputs. Only accepts ONNX models."
        )

        # Model options
        add_input_model_options(sub_parser, enable_onnx=True, default_output_path="optimized-model")

        sub_parser.add_argument(
            "--adapter_format",
            type=str,
            default=WeightsFileFormat.ONNX_ADAPTER,
            choices=[el.value for el in WeightsFileFormat],
            help=f"Format to save the weights in. Default is {WeightsFileFormat.ONNX_ADAPTER}.",
        )

        add_remote_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        sub_parser.set_defaults(func=GenerateAdapterCommand)

    def run(self):
        self._run_workflow()

    def _get_run_config(self, tempdir: str) -> Dict:
        input_model_config = get_input_model_config(self.args)
        assert (
            input_model_config["type"].lower() == "onnxmodel"
        ), "Only ONNX models are supported in generate-adapter command."

        to_replace = [
            ("input_model", input_model_config),
            (("passes", "e", "save_format"), self.args.adapter_format),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]

        config = deepcopy(TEMPLATE)
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)

        update_remote_options(config, self.args, "generate-adapter", tempdir)
        update_shared_cache_options(config, self.args)
        return config


TEMPLATE = {
    "input_model": None,
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": {"e": {"type": "ExtractAdapters"}},
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
