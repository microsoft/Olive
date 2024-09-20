# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from argparse import ArgumentParser
from copy import deepcopy
from typing import ClassVar, Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_logging_options,
    add_model_options,
    add_remote_options,
    get_input_model_config,
    is_remote_run,
    save_output_model,
    update_remote_option,
)
from olive.common.utils import set_nested_dict_value


class GenerateAdapterCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser("generate-adapter", help="Generate ONNX model with adapters as inputs")

        add_logging_options(sub_parser)

        sub_parser.add_argument(
            "--precision",
            type=str,
            default="float16",
            choices=["float16", "float32"],
            help="The precision of the optimized model and adapters.",
        )

        # Model options
        add_model_options(
            sub_parser, enable_hf=True, enable_hf_adapter=True, enable_onnx=True, default_output_path="optimized-model"
        )

        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntie generate() API to run the model"
        )

        sub_parser.add_argument("--clean", action="store_true", help="Run in a clean cache directory")

        # remote options
        add_remote_options(sub_parser)

        sub_parser.set_defaults(func=GenerateAdapterCommand)

    def run(self):
        from olive.workflows import run as olive_run

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self.get_run_config(tempdir)

            olive_run(run_config)

            if is_remote_run(self.args):
                return

            save_output_model(run_config, self.args.output_path)

    def get_run_config(self, tempdir: str) -> Dict:
        from olive.model import ModelConfig

        to_replace = [
            ("input_model", get_input_model_config(self.args)),
            (("input_model", "adapter_path"), self.args.adapter_path),
            (("passes", "o", "float16"), self.args.precision == "float16"),
            # make the mapping of precisions better
            (("passes", "m", "precision"), "fp16" if self.args.precision == "float16" else "fp32"),
            (("clean_cache",), self.args.clean),
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
        ]

        config = deepcopy(TEMPLATE)
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)

        if not self.args.use_ort_genai:
            del config["passes"]["m"]

        update_remote_option(config, self.args, "generate-adapter", tempdir)

        input_model_config = ModelConfig.parse_obj(config["input_model"])
        if input_model_config.type == "hfmodel" and input_model_config.config.get("adapter_path") is None:
            raise ValueError("adapter_path is required for generate-adapter command")
        if input_model_config.type == "onnxmodel":
            del config["passes"]["c"], config["passes"]["o"]

        return config


TEMPLATE = {
    "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            # will just use cuda ep now, only genai metadata is not agnostic to ep
            # revisit once model builder supports lora adapters
            "accelerators": [{"device": "gpu", "execution_providers": ["CUDAExecutionProvider"]}],
        }
    },
    "passes": {
        # TODO(jambayk): migrate to model builder once it supports lora adapters
        # the models produced here are not fully optimized
        "c": {
            "type": "OnnxConversion",
            "target_opset": 17,
            "torch_dtype": "float32",
            "save_metadata_for_token_generation": True,
        },
        "o": {
            "type": "OrtTransformersOptimization",
            "model_type": "gpt2",
            "opt_level": 0,
            "keep_io_types": False,
        },
        "e": {"type": "ExtractAdapters"},
        "m": {"type": "ModelBuilder", "metadata_only": True},
    },
    "host": "local_system",
    "target": "local_system",
}
