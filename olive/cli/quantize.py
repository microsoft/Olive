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

        sub_parser.add_argument(
            "--algorithm",
            type=str,
            required=True,
            choices=sorted(ALGORITHMS.keys()),
            help="List of quantization algorithms to run.",
        )
        sub_parser.add_argument(
            "--precision",
            type=str,
            default="int4",
            choices=["int4", "int8", "int16", "uint4", "uint8", "uint16", "fp4", "fp8", "fp16", "nf4"],
            help="The precision of the quantized model.",
        )
        sub_parser.add_argument(
            "--implementation",
            type=str,
            choices=sorted(TEMPLATE["passes"].keys()),
            help="The specific implementation of quantization algorithms to use.",
        )
        sub_parser.add_argument(
            "--enable-qdq-encoding",
            action="store_true",
            help="Use QDQ encoding in ONNX model for the quantized nodes.",
        )
        sub_parser.add_argument(
            "--quarot_rotate", action="store_true", help="Apply QuaRot/Hadamard rotation to the model."
        )

        add_dataset_options(sub_parser, required=False, include_train=False, include_eval=False)
        add_remote_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        sub_parser.set_defaults(func=QuantizeCommand)

    def _get_run_config(self, tempdir: str) -> Dict[str, Any]:
        config = deepcopy(TEMPLATE)
        update_input_model_options(self.args, config)
        update_dataset_options(self.args, config)
        update_shared_cache_options(config, self.args)

        is_hf_model = config["input_model"]["type"].lower() == "hfmodel"
        if is_hf_model and self.args.algorithm not in ["awq", "gptq", "rtn"]:
            raise ValueError("Selected algorithm is not supported for HuggingFace models.")

        defaults_key = "hf_model_defaults" if is_hf_model else "onnx_model_defaults"

        if not self.args.implementation:
            self.args.implementation = ALGORITHMS[self.args.algorithm][defaults_key]["implementation"]
        if not self.args.precision:
            self.args.precision = ALGORITHMS[self.args.algorithm][defaults_key]["precision"]

        if self.args.algorithm in ["gptq", "rtn"] and self.args.implementation == "quarot":
            self.args.precision = "int16"
        elif self.args.algorithm == "rtn" and self.args.precision == "nf4":
            self.args.implementation = "bnb4"

        if self.args.enable_qdq_encoding and self.args.implementation != "matmul4":
            raise ValueError("QDQ encoding is supported only by matmul4 implementation.")

        if not self.args.implementation or not self.args.precision:
            raise ValueError(
                f"Could not select a valid implementation for algorithm={self.args.algorithm} "
                f"and precision={self.args.precision} combination."
            )

        supported_precisions = IMPLEMENTATIONS[self.args.implementation]["supported_precisions"]
        if supported_precisions and self.args.precision not in supported_precisions:
            raise ValueError(
                f"{IMPLEMENTATIONS[self.args.implementation]['name']} quantizer "
                f"implementation supports only [{', '.join(supported_precisions)}] precisions."
            )

        precision = IMPLEMENTATIONS[self.args.implementation]["precision_mapping"].get(
            self.args.precision, self.args.precision
        )
        if self.args.enable_qdq_encoding and self.args.implementation == "matmul4":
            self.args.implementation = [self.args.implementation, "mnb_to_qdq"]
        else:
            self.args.implementation = [self.args.implementation]

        to_replace = [
            ("pass_flows", [self.args.implementation]),
            (("passes", "awq", "w_bit"), precision),
            (("passes", "gptq", "bits"), precision),
            (("passes", "bnb4", "quant_type"), precision),
            (("passes", "quarot", "w_bits"), precision),
            (("passes", "quarot", "rotate"), self.args.quarot_rotate),
            (("passes", "quarot", "w_rtn"), self.args.algorithm == "rtn"),
            (("passes", "quarot", "w_gptq"), self.args.algorithm == "gptq"),
            (("passes", "nvmo", "precision"), precision),
            (("passes", "nvmo", "algorithm"), self.args.algorithm.upper()),
            (("passes", "onnx_dynamic", "weight_type"), precision),
            (("passes", "inc_dynamic", "algorithm"), self.args.algorithm.upper()),
            (("passes", "inc_dynamic", "bits"), precision),
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

        return config

    def run(self):
        from olive.workflows import run as olive_run

        if ("gptq" in self.args.algorithm) and (not self.args.data_name):
            raise ValueError("data_name is required to use gptq.")

        if ("quarot" in self.args.algorithm) and (not self.args.data_name) and (self.args.quarot_strategy == "gptq"):
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
        "awq": {"type": "AutoAWQQuantizer", "w_bit": 4},
        "gptq": {"type": "GptqQuantizer", "bits": 4, "data_config": "default_data_config"},
        "quarot": {
            "type": "QuaRot",
            "w_bits": 16,
            "w_rtn": False,
            "w_gptq": False,
            "rotate": False,
            "calibration_data_config": "default_data_config",
        },
        # Onnx algorithms
        "bnb4": {"type": "OnnxBnb4Quantization", "quant_type": "nf4"},
        "matmul4": {"type": "OnnxMatMul4Quantizer", "accuracy_level": 4},
        "mnb_to_qdq": {"type": "MatMulNBitsToQDQ"},
        "nvmo": {"type": "NVModelOptQuantization", "precision": "int4", "algorithm": "AWQ"},
        "onnx_dynamic": {"type": "OnnxDynamicQuantization", "weight_type": "QInt8"},
        "inc_dynamic": {"type": "IncDynamicQuantization", "quant_level": "auto", "algorithm": "RTN"},
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
    "awq": {
        "implementations": ["awq", "inc_static", "inc_dynamic"],
        "hf_model_defaults": {"implementation": "awq", "precision": "int4"},
        "onnx_model_defaults": {"implementation": "nvmo", "precision": "int4"},
        "description": "(HfModel, OnnxModel) WOQ with AWQ.",
    },
    "gptq": {
        "implementations": ["gptq", "quarot", "matmul4", "inc_static", "inc_dynamic"],
        "hf_model_defaults": {"implementation": "gptq", "precision": "int4"},
        "onnx_model_defaults": {"implementation": "matmul4", "precision": "int4"},
        "description": "(HfModel, OnnxModel) WOQ with GPTQ.",
    },
    "rtn": {
        "implementations": ["quarot", "bnb4", "matmul4"],
        "hf_model_defaults": {"implementation": "quarot", "precision": "int16"},
        "onnx_model_defaults": {"implementation": "onnx_static", "precision": "int8"},
        "description": "(HfModel, OnnxModel) WOQ with RTN.",
    },
    "hqq": {
        "implementations": ["matmul4"],
        "hf_model_defaults": {"implementation": None, "precision": None},
        "onnx_model_defaults": {"implementation": "matmul4", "precision": "int4"},
        "description": "(OnnxModel) HQQ quantization using onnxruntime.",
    },
    # "static": {
    #     "implementations": ["onnx_static", "inc_static"],
    #     "hf_model_defaults": {"implementation": None, "precision": None},
    #     "onnx_model_defaults": {"implementation": "onnx_static", "precision": "int8"},
    #     "description": "(OnnxModel) Static quantization using onnxruntime.",
    # },
    "dynamic": {
        "implementations": ["onnx_dynamic", "inc_dynamic"],
        "hf_model_defaults": {"implementation": None, "precision": None},
        "onnx_model_defaults": {"implementation": "onnx_dynamic", "precision": "int8"},
        "description": "(OnnxModel) Dynamic quantization using onnxruntime.",
    },
}

IMPLEMENTATIONS = {
    "awq": {
        "name": "WOQ with AWQ",
        "supported_precisions": [],
        "precision_mapping": {
            "int4": 4,
            "int8": 8,
            "int16": 16,
            "uint4": 4,
            "uint8": 8,
            "uint16": 16,
        },
    },
    "gptq": {
        "name": "WOQ with GPTQ",
        "supported_precisions": [],
        "precision_mapping": {
            "int4": 4,
            "int8": 8,
            "int16": 16,
            "uint4": 4,
            "uint8": 8,
            "uint16": 16,
        },
    },
    "quarot": {
        "name": "QuaRot/Hadamard rotation",
        "supported_precisions": [],
        "precision_mapping": {
            "int4": 4,
            "int8": 8,
            "int16": 16,
            "uint4": 4,
            "uint8": 8,
            "uint16": 16,
        },
    },
    "bnb4": {
        "name": "Bits-n-Bytes",
        "supported_precisions": ["fp4", "nf4"],
        "precision_mapping": {},
    },
    "matmul4": {
        "name": "WOQ with MatMulNBits",
        "supported_precisions": ["int4"],
        "precision_mapping": {},
    },
    "nvmo": {
        "name": "nVidia ModelOpt",
        "supported_precisions": ["int4", "int8", "fp8"],
        "precision_mapping": {},
    },
    "onnx_static": {
        "name": "Onnxruntime static",
        "supported_precisions": ["int8", "uint8", "int16", "uint16"],
        "precision_mapping": {
            "int8": "QInt8",
            "uint8": "QUInt8",
            "uint16": "QUInt16",
            "int16": "QInt16",
        },
    },
    "onnx_dynamic": {
        "name": "Onnxruntime dynamic",
        "supported_precisions": ["int8", "uint8"],
        "precision_mapping": {"int8": "QInt8", "uint8": "QUInt8"},
    },
    "inc_static": {
        "name": "Intel® Neural Compressor static",
        "supported_precisions": ["int4"],
        "precision_mapping": {},
    },
    "inc_dynamic": {
        "name": "Intel® Neural Compressor static",
        "supported_precisions": ["int4"],
        "precision_mapping": {},
    },
}
