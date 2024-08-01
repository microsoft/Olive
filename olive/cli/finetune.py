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

from olive.cli.base import BaseOliveCLICommand
from olive.common.utils import hardlink_copy_dir, set_nested_dict_value, set_tempdir


class FineTuneCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "finetune",
            help="Fine-tune a model on a dataset using peft",
        )

        # TODO(jambayk): change to lowercase
        sub_parser.add_argument(
            "--method",
            type=str,
            default="LoRA",
            choices=["LoRA", "QLoRA"],
            help="The method to use for fine-tuning",
        )
        # model options
        model_group = sub_parser.add_argument_group("model options")
        model_group.add_argument(
            "-m",
            "--model_name_or_path",
            type=str,
            required=True,
            help="The model checkpoint for weights initialization.",
        )
        model_group.add_argument(
            "--trust_remote_code", action="store_true", help="Trust remote code when loading a model."
        )
        model_group.add_argument(
            "--torch_dtype", type=str, default="bfloat16", help="The torch dtype to use for training."
        )
        # dataset options
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
            type=str,
            help=r"Template to generate text field from. E.g. '### Question: {prompt} \n### Answer: {response}'",
        )
        dataset_group.add_argument(
            "--max_seq_len",
            type=int,
            default=1024,
            help="Maximum sequence length for the data.",
        )
        # lora options
        lora_group = sub_parser.add_argument_group("lora options")
        lora_group.add_argument(
            "--lora_r",
            type=int,
            default=16,
            help="LoRA R value.",
        )
        lora_group.add_argument(
            "--lora_alpha",
            type=int,
            default=32,
            help="LoRA alpha value.",
        )
        lora_group.add_argument(
            "--target_modules", type=str, help="The target modules for LoRA. If multiple, separate by comma."
        )

        # directory options
        sub_parser.add_argument("-o", "--output_path", type=str, default="finetuned-adapter", help="Output path")
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )

        sub_parser.set_defaults(func=FineTuneCommand)

    def run(self):
        from olive.workflows import run as olive_run

        set_tempdir(self.args.tempdir)

        run_config = self.get_run_config()

        with tempfile.TemporaryDirectory() as tempdir:
            run_config["output_dir"] = tempdir
            olive_run(run_config)

            # need to improve the output structure of olive run
            with open(Path(tempdir) / "finetune" / "gpu-cuda_model.json") as f:
                model_config = json.load(f)

            output_path = Path(self.args.output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            hardlink_copy_dir(model_config["config"]["adapter_path"], output_path)

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

    def get_run_config(self) -> Dict:
        load_key = ("data_configs", 0, "load_dataset_config")
        preprocess_key = ("data_configs", 0, "pre_process_data_config")
        finetune_key = ("passes", "finetune")
        to_replace = [
            (("input_model", "model_path"), self.args.model_name_or_path),
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
        ]
        if self.args.trust_remote_code:
            to_replace.append((("input_model", "load_kwargs", "trust_remote_code"), True))
        if self.args.method == "LoRA":
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
            config["passes"]["finetune"]["eval_data_config"] = "eval_data"

        return config


TEMPLATE = {
    "input_model": {
        "type": "HfModel",
        "load_kwargs": {"attn_implementation": "eager"},
    },
    "systems": {
        "local_system": {
            "type": "LocalSystem",
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
    "passes": {"finetune": {"train_data_config": "train_data"}},
    "host": "local_system",
    "target": "local_system",
}
