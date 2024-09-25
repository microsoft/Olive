# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201
# ruff: noqa: RUF012

import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_dataset_options,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    is_remote_run,
    save_output_model,
    update_dataset_options,
    update_input_model_options,
)
from olive.common.utils import set_nested_dict_value


class QuantizeCommand(BaseOliveCLICommand):
    _CONFIG_TEMPLATE = {
        "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
        "data_configs": [
            {
                "name": "default_data_config",
                "type": "HuggingfaceContainer",
                "load_dataset_config": {},
                "pre_process_data_config": {},
                "dataloader_config": {},
                "post_process_data_config": {},
            }
        ],
        "passes": {
            "awq": {"type": "AutoAWQQuantizer"},
            "gptq": {
                # Ref: https://github.com/AutoGPTQ/AutoGPTQ/pull/651/files
                "type": "GptqQuantizer",
                "data_config": "default_data_config",
            },
        },
        "pass_flows": [],
        "output_dir": "models",
    }

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "quantize",
            help="Quantize the input model",
        )

        # model options
        add_input_model_options(sub_parser, enable_hf=True, enable_pt=True, default_output_path="quantized-model")

        sub_parser.add_argument(
            "--algorithms",
            type=str,
            nargs="*",
            required=True,
            choices=sorted(QuantizeCommand._CONFIG_TEMPLATE["passes"].keys()),
            help="List of quantization algorithms to run.",
        )

        add_dataset_options(sub_parser, required=False, include_train=False, include_eval=False)
        add_remote_options(sub_parser)
        add_logging_options(sub_parser)
        sub_parser.set_defaults(func=QuantizeCommand)

    def _get_run_config(self, tempdir: str) -> Dict[str, Any]:
        config = deepcopy(QuantizeCommand._CONFIG_TEMPLATE)
        update_input_model_options(self.args, config)
        update_dataset_options(self.args, config)

        to_replace = [
            (("pass_flows"), [[name] for name in self.args.algorithms]),
            (("output_dir"), tempdir),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

        return config

    def run(self):
        from olive.workflows import run as olive_run

        if "gptq" in self.args.algorithms and not self.args.data_name:
            raise ValueError("data_name is required to use gptq.")

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self._get_run_config(tempdir)
            olive_run(run_config)

            if is_remote_run(self.args):
                return

            save_output_model(run_config, self.args.output_path)
