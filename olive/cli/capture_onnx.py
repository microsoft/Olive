# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

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
from olive.common.utils import IntEnumBase, hardlink_copy_dir, set_nested_dict_value, set_tempdir


class ModelBuilderAccuracyLevel(IntEnumBase):
    fp32 = 1
    fp16 = 2
    bf16 = 3
    int8 = 4


class CaptureOnnxGraphCommand(BaseOliveCLICommand):
    allow_unknown_args: ClassVar[bool] = True

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "capture-onnx-graph",
            help=("Capture ONNX graph using PyTorch Exporter or Model Builder from the Huggingface model."),
        )

        add_logging_options(sub_parser)

        # model options
        add_hf_model_options(sub_parser)

        sub_parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu"],
            help=(
                "The device to use to convert the model to ONNX."
                "If 'gpu' is selected, the execution_providers will be set to CUDAExecutionProvider."
                "If 'cpu' is selected, the execution_providers will be set to CPUExecutionProvider."
                "For PyTorch Exporter, the device is used to cast the model to before capturing the ONNX graph."
            ),
        )

        sub_parser.add_argument("-o", "--output_path", type=str, default="onnx-model", help="Output path")
        sub_parser.add_argument(
            "--tempdir", default=None, type=str, help="Root directory for tempfile directories and files"
        )

        # PyTorch Exporter options
        pte_group = sub_parser.add_argument_group("PyTorch Exporter options")
        pte_group.add_argument(
            "--use_dynamo_exporter",
            type=bool,
            default=False,
            help="Whether to use dynamo_export API to export ONNX model.",
        )
        pte_group.add_argument(
            "--use_ort_genai", type=bool, default=False, help="Use OnnxRuntie generate() API to run the model"
        )
        pte_group.add_argument(
            "--past_key_value_name",
            type=str,
            default="past_key_values",
            help=(
                "The arguments name to point to past key values. For model loaded from huggingface, "
                "it is 'past_key_values'. Basically, it is used only when `use_dynamo_exporter` is True."
            ),
        )
        pte_group.add_argument(
            "--torch_dtype",
            type=str,
            help=(
                "The dtype to cast the model to before capturing the ONNX graph, e.g., 'float32' or 'float16'."
                " If not specified will use the model as is."
            ),
        )
        pte_group.add_argument(
            "--target_opset",
            type=int,
            default=17,
            help="The target opset version for the ONNX model. Default is 17.",
        )

        # Model Builder options
        mb_group = sub_parser.add_argument_group("Model Builder options")
        mb_group.add_argument(
            "--use_model_builder",
            action="store_true",
            help="Whether to use Model Builder to capture ONNX model.",
        )
        mb_group.add_argument(
            "--precision",
            type=str,
            default="fp16",
            choices=["fp16", "fp32", "int4"],
            help="The precision of the ONNX model. This is used by Model Builder",
        )
        mb_group.add_argument(
            "--int4_block_size",
            type=int,
            required=False,
            choices=[16, 32, 64, 128, 256],
            help="Specify the block_size for int4 quantization. Acceptable values: 16/32/64/128/256.",
        )
        mb_group.add_argument(
            "--int4_accuracy_level",
            type=ModelBuilderAccuracyLevel,
            required=False,
            help="Specify the minimum accuracy level for activation of MatMul in int4 quantization.",
        )
        mb_group.add_argument(
            "--exclude_embeds",
            type=bool,
            default=False,
            required=False,
            help="Remove embedding layer from your ONNX model.",
        )
        mb_group.add_argument(
            "--exclude_lm_head",
            type=bool,
            default=False,
            required=False,
            help="Remove language modeling head from your ONNX model.",
        )
        mb_group.add_argument(
            "--enable_cuda_graph",
            type=bool,
            default=None,  # Explicitly setting to None to differentiate between user intent and default.
            required=False,
            help=(
                "The model can use CUDA graph capture for CUDA execution provider. "
                "If enabled, all nodes being placed on the CUDA EP is the prerequisite "
                "for the CUDA graph to be used correctly."
            ),
        )

        # remote options
        add_remote_options(sub_parser)

        sub_parser.set_defaults(func=CaptureOnnxGraphCommand)

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
                output_path = Path(self.args.output_path)
                output_path.mkdir(parents=True, exist_ok=True)
                pass_name = "m" if self.args.use_model_builder else "c"
                device_name = "gpu-cuda_model" if self.args.device == "gpu" else "cpu-cpu_model"
                hardlink_copy_dir(Path(tempdir) / pass_name / device_name, output_path)
                print("ONNX Model is saved to %s", output_path.resolve())
            else:
                print("Failed to run capture-onnx-graph. Please set the log_level to 1 for more detailed logs.")

    def get_run_config(self, tempdir: str) -> Dict:
        config = deepcopy(TEMPLATE)

        if self.args.task is not None:
            config["input_model"]["task"] = self.args.task

        to_replace = [
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
            (("input_model", "model_path"), get_model_name_or_path(self.args.model_name_or_path)),
            (("input_model", "load_kwargs", "trust_remote_code"), self.args.trust_remote_code),
            (("systems", "local_system", "accelerators", 0, "device"), self.args.device),
            (
                ("systems", "local_system", "accelerators", 0, "execution_providers"),
                ["CPUExecutionProvider"] if self.args.device == "cpu" else ["CUDAExecutionProvider"],
            ),
        ]
        if self.args.use_model_builder:
            del config["passes"]["c"]
            to_replace.extend(
                [
                    (("passes", "m", "precision"), self.args.precision),
                    (("passes", "m", "exclude_embeds"), self.args.exclude_embeds),
                    (("passes", "m", "exclude_lm_head"), self.args.exclude_lm_head),
                    (("passes", "m", "enable_cuda_graph"), self.args.enable_cuda_graph),
                ]
            )
            if self.args.int4_block_size is not None:
                to_replace.append((("passes", "m", "int4_block_size"), self.args.int4_block_size))
            if self.args.int4_accuracy_level is not None:
                to_replace.append((("passes", "m", "int4_accuracy_level"), self.args.int4_accuracy_level))
        else:
            del config["passes"]["m"]
            to_replace.extend(
                [
                    (("passes", "c", "device"), self.args.device if self.args.device == "cpu" else "cuda"),
                    (("passes", "c", "torch_dtype"), self.args.torch_dtype),
                    (("passes", "c", "target_opset"), self.args.target_opset),
                    (("passes", "c", "use_dynamo_exporter"), self.args.use_dynamo_exporter),
                    (("passes", "c", "save_metadata_for_token_generation"), self.args.use_ort_genai),
                ]
            )
            if self.args.use_dynamo_exporter:
                to_replace.append(("passes", "c", "past_key_value_name"), self.args.past_key_value_name)

        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)
        update_remote_option(config, self.args, "capture-onnx-graph", tempdir)

        return config


TEMPLATE = {
    "input_model": {"type": "HfModel", "load_kwargs": {}},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": {
        "c": {
            "type": "OnnxConversion",
        },
        "m": {"type": "ModelBuilder", "metadata_only": False},
    },
    "host": "local_system",
    "target": "local_system",
}
