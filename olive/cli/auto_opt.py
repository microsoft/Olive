# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import json
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path
from typing import ClassVar, Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_hf_model_options,
    add_logging_options,
    add_remote_options,
    get_model_name_or_path,
    get_output_model_number,
    is_remote_run,
    update_remote_option,
)
from olive.common.utils import set_tempdir

# ruff: noqa: T201


EVALUATE_TEMPLATE = {
    "common_evaluator": {
        "metrics": [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.01}},
                ],
                "data_config": "data_config",
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 20}},
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
            help=("Automatically performance optimize input model"),
        )

        add_hf_model_options(sub_parser)
        add_logging_options(sub_parser)

        # model options
        model_attributes_group = sub_parser.add_argument_group("model attributes options")
        model_attributes_group.add_argument(
            "--model_framework",
            type=str,
            default="hf",
            choices=["hf", "onnx", "pytorch"],
            help="model type, choose from HfModel and ONNXModel",
        )

        system_group = sub_parser.add_argument_group("system options")
        system_group.add_argument(
            "--device",
            type=str,
            default=None,
            choices=["cpu", "gpu", "npu"],
            # TODO(anyone): add more devices cpu_spr, vpu, intel_myriad
            help=(
                "Device to use for optimization, choose from cpu and gpu. If not specified,"
                " will deduce from the value of execution providers, CPU/VistisAi for cpu and"
                " CUDA/Tensorrt/Dml for gpu. If both device and execution providers are not specified,"
                " default to cpu device with CPUExecutionProvider."
            ),
        )
        system_group.add_argument(
            "--providers",
            type=str,
            nargs="*",
            choices=["CPU", "CUDA", "Tensorrt", "Dml", "VitisAI", "Qnn"],
            help="List of execution providers to use for optimization",
        )

        # dataset options
        dataset_group = sub_parser.add_argument_group(
            "dataset options, required for some optimization passes like quantization, and evaluation components"
        )
        dataset_group.add_argument(
            "--data_config_path",
            type=str,
            help="Path to the data config file. It allows to customize the data config(json/yaml) for the model.",
        )

        auto_opt_config_group = sub_parser.add_argument_group("auto optimizer options")
        auto_opt_config_group.add_argument(
            "--precisions",
            type=str,
            nargs="*",
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

        search_strategy_group = sub_parser.add_argument_group("search strategy options")
        search_strategy_group.add_argument(
            "--num_samples", type=int, default=5, help="Number of samples for search algorithm"
        )
        search_strategy_group.add_argument("--seed", type=int, default=0, help="Random seed for search algorithm")
        search_strategy_group.add_argument(
            "--search_order",
            type=str,
            default="joint",
            choices=["joint", "pass-by-pass"],
            help="Execution order for search strategy",
        )
        search_strategy_group.add_argument(
            "--search_algorithm",
            type=str,
            default="tpe",
            choices=["exhaustive", "tpe", "random"],
            help="Search algorithm for search strategy",
        )

        # output options
        output_group = sub_parser.add_argument_group("output options")
        output_group.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )
        output_group.add_argument("-o", "--output_path", type=str, default="auto-opt-output", help="Output path")
        # remote options
        add_remote_options(sub_parser)
        sub_parser.set_defaults(func=AutoOptCommand)

    def _get_data_config(self) -> Dict:
        with open(self.args.data_config_path) as f:
            data_config = json.load(f)
            data_config["name"] = "data_config"
            return data_config

    def run(self):
        from olive.workflows import run as olive_run

        set_tempdir(self.args.output_path)
        with tempfile.TemporaryDirectory() as tempdir:
            run_config = self.get_run_config(tempdir)
            output = olive_run(run_config)
            if is_remote_run(self.args):
                # TODO(jambayk): point user to datastore with outputs or download outputs
                # both are not implemented yet
                return
            if get_output_model_number(output) > 0:
                print("Optimized ONNX Model is saved to ", Path(self.args.output_path).resolve())
            else:
                print("No optimized model is generated")

    def resolve_providers(self):
        self.args.providers = self.args.providers or []
        for idx, provider in enumerate(self.args.providers):
            if not provider.endswith("ExecutionProvider"):
                self.args.providers[idx] = f"{provider}ExecutionProvider"

    def get_run_config(self, tempdir) -> Dict:
        self.resolve_providers()
        config = deepcopy(TEMPLATE)

        config["log_severity_level"] = self.args.log_level
        config["input_model"]["model_path"] = get_model_name_or_path(self.args.model_name_or_path)
        config["input_model"]["type"] = f"{self.args.model_framework}model"
        if self.args.task:
            config["input_model"]["task"] = self.args.task
        config["cache_dir"] = Path(tempdir) / "cache"
        config["output_dir"] = self.args.output_path

        device = self.args.device
        if not device:
            device = (
                "gpu"
                if self.args.providers
                and any(p[: -(len("ExecutionProvider"))] in ["CUDA", "Tensorrt", "Dml"] for p in self.args.providers)
                else "cpu"
            )
        providers = self.args.providers or ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
        config["systems"]["local_system"]["accelerators"] = [{"device": device, "execution_providers": providers}]

        excluded_passes = self.args.excluded_passes or ["ModelBuilder", "OrtPerfTuning"]
        if self.args.use_model_builder:
            excluded_passes.remove("ModelBuilder")
            excluded_passes.append("OnnxConversion")
        config["auto_optimizer_config"] = {
            "precisions": self.args.precisions,
            "excluded_passes": excluded_passes,
        }

        config["search_strategy"] = {
            # TODO(anyone): rename execution_order to search_order in search_strategy
            "execution_order": self.args.search_order,
            "search_algorithm": self.args.search_algorithm,
            "num_samples": self.args.num_samples,
            "seed": self.args.seed,
        }

        if self.args.data_config_path:
            data_configs = self._get_data_config()
            config["data_configs"] = [data_configs]
        else:
            del config["evaluators"]
            del config["evaluator"]
            del config["search_strategy"]
        update_remote_option(config, self.args, "auto-opt", tempdir)
        return config
