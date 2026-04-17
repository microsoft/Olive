# -----------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------
import json
from argparse import ArgumentParser
from copy import deepcopy
from pathlib import Path

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    add_shared_cache_options,
    add_telemetry_options,
    get_input_model_config,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.telemetry import action


def _is_local_onnx_model(model_name_or_path) -> bool:
    """Return True if model_name_or_path clearly points to a local ONNX model.

    Resolves without any network calls. Covers:
    - a local .onnx file
    - a local directory containing one or more .onnx files
    - a previous-command output directory whose model_config.json type is OnnxModel
    """
    if not model_name_or_path:
        return False

    model_path = Path(model_name_or_path)
    if not model_path.exists():
        return False

    if model_path.is_file():
        return model_path.suffix == ".onnx"

    if not model_path.is_dir():
        return False

    model_config_path = model_path / "model_config.json"
    if model_config_path.exists():
        try:
            with open(model_config_path) as f:
                model_config = json.load(f)
            return model_config.get("type", "").lower() == "onnxmodel"
        except (OSError, ValueError):
            return False

    return any(model_path.glob("*.onnx"))


class BenchmarkCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("benchmark", help="Evaluate the model using lm-eval.")

        # model options
        add_input_model_options(
            sub_parser, enable_hf=True, enable_hf_adapter=True, enable_pt=True, default_output_path="onnx-model"
        )

        # lm-eval options
        lmeval_group = sub_parser.add_argument_group("lm-eval evaluator options")
        lmeval_group.add_argument(
            "--tasks",
            type=str,
            required=True,
            nargs="*",
            help="List of tasks to evaluate on.",
        )

        lmeval_group.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu"],
            help="Target device for evaluation.",
        )

        lmeval_group.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size.",
        )

        lmeval_group.add_argument(
            "--max_length",
            type=int,
            default=1024,
            help="Maximum length of input + output.",
        )

        lmeval_group.add_argument(
            "--limit",
            type=float,
            default=1,
            help="Number (or percentage of dataset) of samples to use for evaluation.",
        )

        lmeval_group.add_argument(
            "--backend",
            type=str,
            default="auto",
            choices=["auto", "ort", "ortgenai"],
            help="Backend for ONNX model evaluation. Use 'auto' to infer backend from model type.",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_telemetry_options(sub_parser)
        sub_parser.set_defaults(func=BenchmarkCommand)

    @action
    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict:
        config = deepcopy(TEMPLATE)

        # Short-circuit: validate --backend against local inputs before
        # get_input_model_config, which may trigger a network call
        # (hf_repo_exists) for unknown model ids.
        if self.args.backend != "auto" and not _is_local_onnx_model(self.args.model_name_or_path):
            raise ValueError("--backend is only supported for ONNX input models.")

        input_model_config = get_input_model_config(self.args)
        assert input_model_config["type"].lower() in {
            "hfmodel",
            "pytorchmodel",
            "onnxmodel",
        }, "Only HfModel, PyTorchModel and OnnxModel are supported in benchmark command."

        if self.args.backend != "auto" and input_model_config["type"].lower() != "onnxmodel":
            raise ValueError("--backend is only supported for ONNX input models.")

        to_replace = [
            ("input_model", input_model_config),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
            (("systems", "local_system", "accelerators", 0, "device"), self.args.device),
            (
                ("systems", "local_system", "accelerators", 0, "execution_providers"),
                [("CUDAExecutionProvider" if self.args.device == "gpu" else "CPUExecutionProvider")],
            ),
            (("evaluators", "evaluator", "tasks"), self.args.tasks),
            (("evaluators", "evaluator", "device"), self.args.device),
            (("evaluators", "evaluator", "batch_size"), self.args.batch_size),
            (("evaluators", "evaluator", "max_length"), self.args.max_length),
            (("evaluators", "evaluator", "device"), self.args.device),
            (("evaluators", "evaluator", "limit"), self.args.limit),
            (
                ("evaluators", "evaluator", "model_class"),
                None if self.args.backend == "auto" else self.args.backend,
            ),
        ]

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)
        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "evaluators": {
        "evaluator": {
            "type": "LMEvaluator",
            "tasks": [],
            "batch_size": 16,
            "max_length": 1024,
            "device": "cpu",
            "limit": 64,
        }
    },
    "evaluator": "evaluator",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
