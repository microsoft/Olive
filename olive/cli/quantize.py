# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201
# ruff: noqa: RUF012

from argparse import ArgumentParser
from copy import deepcopy
from typing import Any, Dict

from olive.cli.base import (
    BaseOliveCLICommand,
    add_dataset_options,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_save_config_file_options,
    add_shared_cache_options,
    update_dataset_options,
    update_input_model_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.package_config import OlivePackageConfig


class QuantizeCommand(BaseOliveCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "quantize",
            help="Quantize the input model",
        )

        # model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="quantized-model",
        )

        sub_parser.add_argument(
            "--algorithm",
            type=str,
            default="rtn",
            choices=["awq", "gptq", "rtn", "hqq"],
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
            # choices=sorted(TEMPLATE["passes"].keys()),
            help="The specific implementation of quantization algorithms to use.",
        )
        sub_parser.add_argument(
            "--use_qdq_encoding",
            action="store_true",
            help="Use QDQ encoding in ONNX model for the quantized nodes.",
        )

        add_dataset_options(sub_parser, required=False, include_train=False, include_eval=False)
        add_remote_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        sub_parser.set_defaults(func=QuantizeCommand)

    def _get_precision_bits(self):
        PRECISION_TO_BITS = {
            "int4": 4,
            "int8": 8,
            "int16": 16,
            "uint4": 4,
            "uint8": 8,
            "uint16": 16,
        }
        return PRECISION_TO_BITS[self.args.precision]

    def _get_precision_in_wtypes(self):
        PRECISION_TO_WTYPES = {
            "int8": "QInt8",
            "uint8": "QUInt8",
            "int16": "QInt16",
            "uint16": "QUInt16",
        }
        return PRECISION_TO_WTYPES[self.args.precision]

    def _build_pass_list(self, P, A, I, is_hf_model):
        olive_config = OlivePackageConfig.load_default_config()
        pass_list = []
        available_passes_list = PT_QUANT_IMPLEMENTATION_MAPPING
        if not is_hf_model:
            available_passes_list = ONNX_QUANT_IMPLEMENTATION_MAPPING
        for r in available_passes_list:
            pinfo = olive_config.get_pass_module_config(r["implementation_class"])
            if I is None or r["name"] == I:
                if A is None or A in pinfo.supported_algorithms:
                    if P is None or P in pinfo.supported_precisions:
                        if not self.args.use_qdq_encoding or "qdq" in pinfo.supported_quantization_encodings:
                            if (pinfo.dataset_required and self.args.data_name) or (not pinfo.dataset_required):
                                pass_list.append(r)

        print(f"_build_pass_list {pass_list}")
        return pass_list

    def _get_passes_dict(self, plist):
        precision_in_bits = self._get_precision_bits()
        wtypes = self._get_precision_in_wtypes()
        quant_format = "QDQ"
        if self.args.use_qdq_encoding:
            quant_format = "QOP"

        # config options to add for a given option
        to_add = {
            "awq": {"w_bits": precision_in_bits},
            "gptq": {"bits", precision_in_bits},
            "bnb4": {"quant_type": precision_in_bits},
            "nvmo": {"precision": precision_in_bits, "algorithm": self.args.algorithm.upper()},
            "OnnxDynamicQuantization": {"weight_type": wtypes, "quant_format": quant_format},
            "matmul4": {"quant_format": quant_format},
            "inc_dynamic": {"algorithm": self.args.algorithm.upper(), "bits": precision_in_bits},
        }

        nplist = {}
        for p in plist:
            pt = p["implementation_class"]
            pd = {"type": pt}
            if to_add.get(pt) is not None:
                for k, v in to_add[pt].items():
                    pd[k] = v
            nplist[pt.lower()] = pd
        print(nplist)
        return nplist

    def _customize_config(self, config):
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

    def _get_run_config(self, tempdir: str) -> Dict[str, Any]:
        config = deepcopy(TEMPLATE)
        update_input_model_options(self.args, config)
        update_dataset_options(self.args, config)
        update_shared_cache_options(config, self.args)

        is_hf_model = config["input_model"]["type"].lower() == "hfmodel"

        # Build list of quantization passes to run
        plist = self._build_pass_list(self.args.precision, self.args.algorithm, self.args.implementation, is_hf_model)

        # Get the passes dictionary for the config
        config["passes"] = self._get_passes_dict(plist)

        # Customize the config for user choices
        self._customize_config(config)
        return config

    def run(self):
        self._run_workflow()


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
        "gptq": {"type": "GptqQuantizer", "bits": 4},
        # Onnx algorithms
        "bnb4": {"type": "OnnxBnb4Quantization", "quant_type": "nf4"},
        "matmul4": {"type": "OnnxMatMul4Quantizer", "accuracy_level": 4},
        "mnb_to_qdq": {"type": "MatMulNBitsToQDQ"},
        "nvmo": {"type": "NVModelOptQuantization", "precision": "int4", "algorithm": "AWQ"},
        "onnx_dynamic": {"type": "OnnxDynamicQuantization", "weight_type": "QInt8"},
        "inc_dynamic": {"type": "IncDynamicQuantization", "quant_level": "auto", "algorithm": "RTN"},
        "onnx_static": {"type": "OnnxStaticQuantization", "data_config": "default_data_config"},
        "inc_static": {"type": "IncStaticQuantization", "data_config": "default_data_config"},
        # "vitis": {"type": "VitisAIQuantization", "data_config": "default_data_config"},
    },
    "output_dir": "models",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}

PT_QUANT_IMPLEMENTATION_MAPPING = [
    {"name": "awq", "implementation_class": "AutoAWQQuantizer"},
    {"name": "autogptq", "implementation_class": "GptqQuantizer"},
]

ONNX_QUANT_IMPLEMENTATION_MAPPING = [
    {"name": "bnb", "implementation_class": "OnnxBnB4Quantization"},
    {"name": "ort", "implementation_class": "OnnxMatMul4Quantizer"},
    {"name": "ort", "implementation_class": "OnnxDynamicQuantization"},
    {"name": "ort", "implementation_class": "OnnxstaticQuantization"},
    {"name": "nvmo", "implementation_class": "NVModelOptQuantization"},
    {"name": "inc", "implementation_class": "IncDynamicQuantization"},
    {"name": "inc", "implementation_class": "IncStaticQuantization"},
]
