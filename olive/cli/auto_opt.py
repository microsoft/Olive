# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from typing import ClassVar, Dict, List

from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_search_options,
    get_input_model_config,
    is_remote_run,
    save_output_model,
    update_accelerator_options,
    update_remote_options,
    update_search_options,
)
from olive.common.utils import hardlink_copy_dir, set_nested_dict_value

EVALUATE_TEMPLATE = {
    "common_evaluator": {
        "metrics": [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.1}},
                ],
                "data_config": "data_config",
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 1}},
                ],
                "data_config": "data_config",
                "user_config": {"io_bind": True},
            },
        ]
    }
}

TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "auto_optimizer_config": {},
    "search_strategy": {"execution_order": "joint", "search_algorithm": "tpe", "num_samples": 5, "seed": 0},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "host": "local_system",
    "evaluators": EVALUATE_TEMPLATE,
    "evaluator": "common_evaluator",
    "target": "local_system",
}


class AutoOptCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "auto-opt",
            help="Automatically optimize the performance of the input model.",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="auto-opt-output",
        )

        # add accelerator options
        add_accelerator_options(sub_parser)

        # dataset options
        dataset_group = sub_parser.add_argument_group()
        dataset_group.add_argument(
            "-d",
            "--data_name",
            type=str,
            help="The dataset name.",
        )
        dataset_group.add_argument(
            "--split",
            type=str,
            help="The dataset split to use for evaluation.",
        )
        dataset_group.add_argument(
            "--subset",
            type=str,
            help="The dataset subset to use for evaluation.",
        )
        dataset_group.add_argument(
            "--input_cols",
            type=str,
            nargs="*",
            help="The input columns to use for evaluation.",
        )
        dataset_group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size for evaluation.",
        )

        auto_opt_config_group = sub_parser.add_argument_group("auto optimizer options")
        auto_opt_config_group.add_argument(
            "--precision",
            type=str,
            choices=["fp16", "fp32", "int4", "int8"],
            help=(
                "The output precision of the optimized model. If not specified, "
                "the default precision is fp32 for cpu and fp16 for gpu"
            ),
        )
        auto_opt_config_group.add_argument(
            "--excluded_passes",
            type=str,
            nargs="*",
            help=(
                "List of passes to disable for optimization, if not specified, "
                "auto-opt will disable ModelBuilder/OrtPerfTuning by default."
            ),
        )
        auto_opt_config_group.add_argument(
            "--use_model_builder",
            action="store_true",
            help=(
                "Whether to use model builder pass for optimization, enable only "
                "when the model is supported by model builder"
            ),
        )

        # search options
        add_search_options(sub_parser)

        # remote options
        add_remote_options(sub_parser)
        add_logging_options(sub_parser)
        sub_parser.set_defaults(func=AutoOptCommand)

    def run(self):
        from olive.workflows import run as olive_run

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self.get_run_config(tempdir)

            olive_run(run_config)

            if is_remote_run(self.args):
                return

            if run_config.get("search_strategy"):
                # TODO(anyone): maybe save the best model instead of just the search results
                hardlink_copy_dir(run_config["output_dir"], self.args.output_path)
                print(f"Search results are saved to {self.args.output_path}")
            else:
                save_output_model(run_config, self.args.output_path)

    def get_run_config(self, tempdir) -> Dict:
        config = deepcopy(TEMPLATE)

        excluded_passes = self.args.excluded_passes or ["ModelBuilder", "OrtPerfTuning"]
        if self.args.use_model_builder:
            excluded_passes.remove("ModelBuilder")
            excluded_passes.append("OnnxConversion")

        to_replace = [
            ("input_model", get_input_model_config(self.args)),
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
            ("data_configs", self._get_data_config()),
            ("auto_optimizer_config", {"precisions": [self.args.precision], "excluded_passes": excluded_passes}),
        ]
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)

        update_accelerator_options(self.args, config)
        update_search_options(self.args, config)
        update_remote_options(config, self.args, "auto-opt", tempdir)

        if self.args.enable_search is None:
            del config["evaluators"]
            del config["evaluator"]
            del config["search_strategy"]
        elif not config["data_configs"]:
            raise ValueError("Dataset is required when search is enabled")

        return config

    def _get_data_config(self) -> List[Dict]:
        if not self.args.data_name:
            return []

        to_replace = [
            (("load_dataset_config", "data_name"), self.args.data_name),
            (("load_dataset_config", "split"), self.args.split),
            (("load_dataset_config", "subset"), self.args.subset),
            (("pre_process_data_config", "input_cols"), self.args.input_cols),
            (("dataloader_config", "batch_size"), self.args.batch_size),
        ]
        data_config = {
            "name": "data_config",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {},
            "pre_process_data_config": {},
            "dataloader_config": {},
        }
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(data_config, keys, value)

        return [data_config]
