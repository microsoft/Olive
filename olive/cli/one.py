# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    get_input_model_config,
)


class OneCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "one",
            help="Run a single pass on the input model",
        )

        # Pass selection
        sub_parser.add_argument(
            "--pass-name",
            type=str,
            required=True,
            help="Name of the pass to run on the input model.",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="one-output",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        sub_parser.set_defaults(func=OneCommand)

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
        # Import these only when needed to avoid circular imports
        from olive.common.utils import set_nested_dict_value
        from olive.package_config import OlivePackageConfig
        
        config = deepcopy(TEMPLATE)
        
        # Set input model from args
        config["input_model"] = get_input_model_config(self.args)
        
        # Set the single pass configuration
        pass_name = self.args.pass_name
        
        # Validate that the pass exists
        olive_config = OlivePackageConfig.load_default_config()
        try:
            olive_config.get_pass_module_config(pass_name)
        except ValueError:
            available_passes = list(olive_config.passes.keys())
            raise ValueError(
                f"Pass '{pass_name}' not found. Available passes: {', '.join(available_passes)}"
            )
        
        # Create a simple pass configuration
        config["passes"] = {
            pass_name.lower(): {
                "type": pass_name
            }
        }
        
        # Customize the config for user choices
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)
        
        return config

    def run(self):
        self._run_workflow()


# Template configuration for the one command
TEMPLATE = {
    "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "output_dir": "models",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}