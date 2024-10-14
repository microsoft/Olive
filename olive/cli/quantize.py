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
    add_shared_cache_options,
    is_remote_run,
    save_output_model,
    update_dataset_options,
    update_input_model_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value


class QuantizeCommand(BaseOliveCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "quantize",
            help="Quantize the input model",
        )

        # model options
        add_input_model_options(sub_parser, enable_hf=True, enable_pt=True, default_output_path="quantized-model")

        newline = "\n"
        sub_parser.add_argument(
            "--algorithms",
            type=str,
            nargs="*",
            required=True,
            choices=list(ALGORITHMS.keys()),
            help=(
                "List of quantization algorithms to"
                f" run.{newline}{newline.join([f'{k}: {v[1]}' for k, v in ALGORITHMS.items()])}"
            ),
        )

        add_dataset_options(sub_parser, required=False, include_train=False, include_eval=False)

        sub_parser.add_argument(
            "--quarot_rotate", action="store_true", help="Apply QuaRot/Hadamard rotation to the model."
        )
        sub_parser.add_argument(
            "--quarot_strategy",
            type=str,
            choices=["rtn", "gptq"],
            default="rtn",
            help="Strategy to use to quantize weights when using QuaRot.",
        )

        add_remote_options(sub_parser)
        add_logging_options(sub_parser)
        add_shared_cache_options(sub_parser)
        sub_parser.set_defaults(func=QuantizeCommand)

    def _get_run_config(self, tempdir: str) -> Dict[str, Any]:
        config = deepcopy(TEMPLATE)
        update_input_model_options(self.args, config)
        update_dataset_options(self.args, config)
        update_shared_cache_options(config, self.args)

        to_replace = [
            ("pass_flows", [deepcopy(ALGORITHMS[algo][0]) for algo in self.args.algorithms]),
            (("passes", "quarot", "rotate"), self.args.quarot_rotate),
            (("passes", "quarot", "w_rtn"), self.args.quarot_strategy == "rtn"),
            (("passes", "quarot", "w_gptq"), self.args.quarot_strategy == "gptq"),
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

        return config

    def run(self):
        from olive.workflows import run as olive_run

        if ("gptq" in self.args.algorithms) and (not self.args.data_name):
            raise ValueError("data_name is required to use gptq.")

        if ("quarot" in self.args.algorithms) and (not self.args.data_name) and (self.args.quarot_strategy == "gptq"):
            raise ValueError("data_name is required to quantize weights using gptq.")

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self._get_run_config(tempdir)
            olive_run(run_config)

            if is_remote_run(self.args):
                return

            save_output_model(run_config, self.args.output_path)


TEMPLATE = {
    "input_model": {"type": "HfModel", "load_kwargs": {"attn_implementation": "eager"}},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
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
        # Pytorch algorithms
        "awq": {"type": "AutoAWQQuantizer"},
        "gptq": {"type": "GptqQuantizer", "data_config": "default_data_config"},
        "quarot": {
            "type": "QuaRot",
            "w_rtn": False,
            "w_gptq": False,
            "rotate": False,
            "calibration_data_config": "default_data_config",
        },
        # Onnx algorithms
        "bnb4": {"type": "OnnxBnb4Quantization"},
        "matmul4": {"type": "OnnxMatMul4Quantizer"},
        "mnb_to_qdq": {"type": "MatMulNBitsToQDQ"},
        "nvmo": {"type": "NVModelOptQuantization"},
        "onnx_dynamic": {"type": "OnnxDynamicQuantization"},
        "inc_dynamic": {"type": "IncDynamicQuantization"},
        # NOTE(all): Not supported yet!
        # "onnx_static": {"type": "OnnxStaticQuantization", "data_config": "default_data_config"},
        # "inc_static": {"type": "IncStaticQuantization", "data_config": "default_data_config"},
        # "vitis": {"type": "VitisAIQuantization", "data_config": "default_data_config"},
    },
    "pass_flows": [],
    "output_dir": "models",
    "host": "local_system",
    "target": "local_system",
}

ALGORITHMS = {
    "awq": (["awq"], "(HfModel) int4 WOQ with AWQ."),
    "gptq": (["gptq"], "(HfModel) int4 WOQ with GPTQ."),
    "quarot": (["quarot"], "(HfModel) QuaRot/Hadamard rotation + AWQ/RTN quantization."),
    "bnb4": (["bnb4"], "(OnnxModel) nf4 WOQ using bitsandbytes."),
    "int4_rtn": (["matmul4", "mnb_to_qdq"], "(OnnxModel) int4 WOQ with RTN using onnxruntime. DQ->MatMul is used."),
    "int4_rtn_mnb": (
        ["matmul4"],
        "(OnnxModel) int4 WOQ with RTN using onnxruntime. MatMulNBits contrib is used instead of DQ->MatMul.",
    ),
    "nvmo": (["nvmo"], "(OnnxModel) int4 WOQ with RTN using Nvidia-ModelOpt. DQ->MatMul is used."),
    "onnx_dynamic": (["onnx_dynamic"], "(OnnxModel) Dynamic quantization using onnxruntime."),
    "inc_dynamic": (["inc_dynamic"], "(OnnxModel) Dynamic quantization using neural-compressor."),
}
