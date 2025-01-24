# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_save_config_file_options,
    add_shared_cache_options,
    get_input_model_config,
    update_remote_options,
    update_shared_cache_options,
)
from olive.common.utils import IntEnumBase, set_nested_dict_value


class ModelBuilderAccuracyLevel(IntEnumBase):
    fp32 = 1
    fp16 = 2
    bf16 = 3
    int8 = 4


class CaptureOnnxGraphCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "capture-onnx-graph",
            help=(
                "Capture ONNX graph using PyTorch Exporter or Model Builder "
                "from the Huggingface model or PyTorch model."
            ),
        )

        # model options
        add_input_model_options(
            sub_parser, enable_hf=True, enable_hf_adapter=True, enable_pt=True, default_output_path="onnx-model"
        )

        sub_parser.add_argument(
            "--conversion_device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu"],
            help="The device used to run the model to capture the ONNX graph.",
        )

        # PyTorch Exporter options
        pte_group = sub_parser.add_argument_group("PyTorch Exporter options")
        pte_group.add_argument(
            "--use_dynamo_exporter",
            action="store_true",
            help="Whether to use dynamo_export API to export ONNX model.",
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

        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntime generate() API to run the model"
        )

        # remote options
        add_remote_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        add_shared_cache_options(sub_parser)
        sub_parser.set_defaults(func=CaptureOnnxGraphCommand)

    def run(self):
        self._run_workflow()

    def _get_run_config(self, tempdir: str) -> Dict:
        config = deepcopy(TEMPLATE)

        input_model_config = get_input_model_config(self.args)
        assert input_model_config["type"].lower() in {
            "hfmodel",
            "pytorchmodel",
        }, "Only HfModel and PyTorchModel are supported in capture-onnx-graph command."

        # whether model is in fp16 (currently not supported by CPU EP)
        is_fp16 = (not self.args.use_model_builder and self.args.torch_dtype == "float16") or (
            self.args.use_model_builder and self.args.precision == "fp16"
        )
        to_replace = [
            ("input_model", input_model_config),
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
            (("systems", "local_system", "accelerators", 0, "device"), "gpu" if is_fp16 else "cpu"),
            (
                ("systems", "local_system", "accelerators", 0, "execution_providers"),
                [("CUDAExecutionProvider" if is_fp16 else "CPUExecutionProvider")],
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
            to_replace.extend(
                [
                    (
                        ("passes", "c", "device"),
                        self.args.conversion_device if self.args.conversion_device == "cpu" else "cuda",
                    ),
                    (("passes", "c", "torch_dtype"), self.args.torch_dtype),
                    (("passes", "c", "target_opset"), self.args.target_opset),
                    (("passes", "c", "use_dynamo_exporter"), self.args.use_dynamo_exporter),
                    (("passes", "c", "save_metadata_for_token_generation"), self.args.use_ort_genai),
                ]
            )
            if self.args.use_dynamo_exporter:
                to_replace.append((("passes", "c", "past_key_value_name"), self.args.past_key_value_name))
            if not self.args.use_ort_genai:
                del config["passes"]["m"]
            else:
                to_replace.extend(
                    [
                        (("passes", "m", "precision"), "fp16" if is_fp16 else "fp32"),
                        (("passes", "m", "metadata_only"), True),
                    ]
                )

        for keys, value in to_replace:
            if value is None:
                continue
            set_nested_dict_value(config, keys, value)
        update_remote_options(config, self.args, "capture-onnx-graph", tempdir)
        update_shared_cache_options(config, self.args)

        return config


TEMPLATE = {
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            # might need an ep option to set for model builder, it is sensitive to ep
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
    "no_artifacts": True,
}
