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
    add_hf_model_options,
    add_logging_options,
    add_remote_options,
    get_model_name_or_path,
    get_output_model_number,
    is_remote_run,
    update_remote_option,
)
from olive.common.utils import hardlink_copy_dir, set_nested_dict_value, set_tempdir, unescaped_str

logger = logging.getLogger(__name__)


class FineTuneCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "finetune",
            help=(
                "Fine-tune a model on a dataset using peft and optimize the model for ONNX Runtime with adapters as"
                " inputs. Huggingface training arguments can be provided along with the defined options."
            ),
        )

        add_logging_options(sub_parser)

        # TODO(jambayk): option to list/install required dependencies?
        sub_parser.add_argument(
            "--precision",
            type=str,
            default="float16",
            choices=["float16", "float32"],
            help="The precision of the optimized model and adapters.",
        )

        # Model options
        add_hf_model_options(sub_parser)

        sub_parser.add_argument(
            "--torch_dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float16", "float32"],
            help="The torch dtype to use for training.",
        )
        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntie generate() API to run the model"
        )

        # Dataset options
        dataset_group = sub_parser.add_argument_group("dataset options")
        dataset_group.add_argument(
            "-d",
            "--data_name",
            type=str,
            required=True,
            help="The dataset name.",
        )
        # TODO(jambayk): currently only supports single file or list of files, support mapping
        dataset_group.add_argument(
            "--data_files", type=str, help="The dataset files. If multiple files, separate by comma."
        )
        dataset_group.add_argument("--train_split", type=str, default="train", help="The split to use for training.")
        dataset_group.add_argument(
            "--eval_split",
            default="",
            help="The dataset split to evaluate on.",
        )
        text_group = dataset_group.add_mutually_exclusive_group(required=True)
        text_group.add_argument(
            "--text_field",
            type=str,
            help="The text field to use for fine-tuning.",
        )
        text_group.add_argument(
            "--text_template",
            # using special string type to allow for escaped characters like \n
            type=unescaped_str,
            help=r"Template to generate text field from. E.g. '### Question: {prompt} \n### Answer: {response}'",
        )
        dataset_group.add_argument(
            "--max_seq_len",
            type=int,
            default=1024,
            help="Maximum sequence length for the data.",
        )
        # LoRA options
        lora_group = sub_parser.add_argument_group("lora options")
        lora_group.add_argument(
            "--method",
            type=str,
            default="lora",
            choices=["lora", "qlora"],
            help="The method to use for fine-tuning",
        )
        lora_group.add_argument(
            "--lora_r",
            type=int,
            default=64,
            help="LoRA R value.",
        )
        lora_group.add_argument(
            "--lora_alpha",
            type=int,
            default=16,
            help="LoRA alpha value.",
        )
        # peft doesn't know about phi3, should we set it ourself in the lora pass based on model type?
        lora_group.add_argument(
            "--target_modules", type=str, help="The target modules for LoRA. If multiple, separate by comma."
        )

        # directory options
        sub_parser.add_argument("-o", "--output_path", type=str, default="optimized-model", help="Output path")
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )
        # TODO(jambayk): what about checkpoint_dir and resume from checkpoint support? clean checkpoint dir?
        sub_parser.add_argument("--clean", action="store_true", help="Run in a clean cache directory")

        # remote options
        add_remote_options(sub_parser)

        sub_parser.set_defaults(func=FineTuneCommand)

    def run(self):
        from olive.workflows import run as olive_run

        set_tempdir(self.args.tempdir)

        with tempfile.TemporaryDirectory() as tempdir:
            run_config = self.get_run_config(tempdir)

            output = olive_run(run_config)

            if is_remote_run(self.args):
                # TODO(jambayk): point user to datastore with outputs or download outputs
                # both are not implemented yet
                return

            if get_output_model_number(output) > 0:
                # need to improve the output structure of olive run
                output_path = Path(self.args.output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                hardlink_copy_dir(Path(tempdir) / "-".join(run_config["passes"].keys()) / "gpu-cuda_model", output_path)
                logger.info("Model and adapters saved to %s", output_path.resolve())
            else:
                logger.error("Failed to run finetune. Please set the log_level to 1 for more detailed logs.")

    def parse_training_args(self) -> Dict:
        if not self.unknown_args:
            return {}

        from transformers import HfArgumentParser, TrainingArguments

        arg_keys = {el[2:] for el in self.unknown_args if el.startswith("--")}
        parser = HfArgumentParser(TrainingArguments)
        # output_dir is required by the parser
        training_args = parser.parse_args(
            [*(["--output_dir", "dummy"] if "output_dir" not in arg_keys else []), *self.unknown_args]
        )

        return {k: v for k, v in vars(training_args).items() if k in arg_keys}

    def get_run_config(self, tempdir: str) -> Dict:
        load_key = ("data_configs", 0, "load_dataset_config")
        preprocess_key = ("data_configs", 0, "pre_process_data_config")
        finetune_key = ("passes", "f")
        model_path = get_model_name_or_path(self.args.model_name_or_path)
        to_replace = [
            (("input_model", "model_path"), model_path),
            ((*load_key, "data_name"), self.args.data_name),
            ((*load_key, "split"), self.args.train_split),
            (
                (*load_key, "data_files"),
                self.args.data_files.split(",") if self.args.data_files else None,
            ),
            ((*preprocess_key, "text_cols"), self.args.text_field),
            ((*preprocess_key, "text_template"), self.args.text_template),
            ((*preprocess_key, "max_seq_len"), self.args.max_seq_len),
            ((*finetune_key, "type"), self.args.method),
            ((*finetune_key, "torch_dtype"), self.args.torch_dtype),
            ((*finetune_key, "training_args"), self.parse_training_args()),
            ((*finetune_key, "lora_r"), self.args.lora_r),
            ((*finetune_key, "lora_alpha"), self.args.lora_alpha),
            (("passes", "o", "float16"), self.args.precision == "float16"),
            # make the mapping of precisions better
            (("passes", "m", "precision"), "fp16" if self.args.precision == "float16" else "fp32"),
            (("clean_cache",), self.args.clean),
            ("output_dir", tempdir),
        ]
        if self.args.trust_remote_code:
            to_replace.append((("input_model", "load_kwargs", "trust_remote_code"), True))
        if self.args.method == "lora" and self.args.target_modules:
            to_replace.append(((*finetune_key, "target_modules"), self.args.target_modules.split(",")))

        config = deepcopy(TEMPLATE)
        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)

        if self.args.eval_split:
            eval_data_config = deepcopy(config["data_configs"][0])
            eval_data_config["name"] = "eval_data"
            eval_data_config["load_dataset_config"]["split"] = self.args.eval_split
            config["data_configs"].append(eval_data_config)
            config["passes"]["f"]["eval_data_config"] = "eval_data"

        if not self.args.use_ort_genai:
            del config["passes"]["m"]

        update_remote_option(config, self.args, "finetune", tempdir)
        config["log_severity_level"] = self.args.log_level

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
    "data_configs": [
        {
            "name": "train_data",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {},
            "pre_process_data_config": {},
        }
    ],
    "passes": {
        "f": {"train_data_config": "train_data"},
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

AZUREML_SYSTEM_TEMPLATE = {
    "type": "AzureML",
    "accelerators": [{"device": "GPU", "execution_providers": ["CUDAExecutionProvider"]}],
    "aml_docker_config": {"base_image": "mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04"},
}
