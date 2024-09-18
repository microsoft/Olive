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
from olive.common.utils import set_nested_dict_value, set_tempdir, unescaped_str


class FineTuneCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "finetune",
            help=(
                "Fine-tune a model on a dataset using peft. Huggingface training arguments can be provided along with"
                " the defined options."
            ),
        )

        add_logging_options(sub_parser)

        # Model options
        add_model_options(sub_parser, enable_hf=True)

        sub_parser.add_argument(
            "--torch_dtype",
            type=str,
            default="bfloat16",
            choices=["bfloat16", "float16", "float32"],
            help="The torch dtype to use for training.",
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
        sub_parser.add_argument("-o", "--output_path", type=str, default="finetuned-adapter", help="Output path")
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

            olive_run(run_config)

            if is_remote_run(self.args):
                return

            save_output_model(run_config, self.args.output_path)

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
        to_replace = [
            ("input_model", get_input_model_config(self.args)),
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
            (("clean_cache",), self.args.clean),
            ("output_dir", tempdir),
        ]
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

        update_remote_option(config, self.args, "finetune", tempdir)
        config["log_severity_level"] = self.args.log_level

        return config


TEMPLATE = {
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
    "passes": {"f": {"train_data_config": "train_data"}},
    "host": "local_system",
    "target": "local_system",
}
