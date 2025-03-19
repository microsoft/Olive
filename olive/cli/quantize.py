# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201
# ruff: noqa: RUF012

import logging
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
from olive.constants import QuantAlgorithm
from olive.package_config import OlivePackageConfig

logger = logging.getLogger(__name__)


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
            choices=[a.value for a in QuantAlgorithm],
            help="List of quantization algorithms to run.",
        )
        sub_parser.add_argument(
            "--precision",
            type=str,
            default="int8",
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
        precision_to_bits = {
            "int4": 4,
            "int8": 8,
            "int16": 16,
            "uint4": 4,
            "uint8": 8,
            "uint16": 16,
        }
        return precision_to_bits.get(self.args.precision)

    def _get_precision_in_wtypes(self):
        precision_to_wtypes = {
            "int8": "QInt8",
            "uint8": "QUInt8",
            "int16": "QInt16",
            "uint16": "QUInt16",
        }
        return precision_to_wtypes.get(self.args.precision)

    def _check_data_name_arg(self, pinfo):
        from olive.constants import DatasetRequirement

        if pinfo.dataset == DatasetRequirement.OPTIONAL:
            return True
        if pinfo.dataset == DatasetRequirement.REQUIRED and self.args.data_name:
            return True
        return pinfo.dataset == DatasetRequirement.NOT_REQUIRED and not self.args.data_name

    def _get_pass_list(self, precision, algo, impl, is_hf_model):
        olive_config = OlivePackageConfig.load_default_config()
        pass_list = []
        available_passes_list = PT_QUANT_IMPLEMENTATION_MAPPING
        if not is_hf_model:
            available_passes_list = ONNX_QUANT_IMPLEMENTATION_MAPPING
        for r in available_passes_list:
            pinfo = olive_config.get_pass_module_config(r["pass_type"])
            if (
                (impl is None or r["impl_name"] == impl)  # pylint: disable=R0916
                and (algo is None or algo in pinfo.supported_algorithms)
                and (precision is None or precision in pinfo.supported_precisions)
                and (not self.args.use_qdq_encoding or "qdq" in pinfo.supported_quantization_encodings)
                and self._check_data_name_arg(pinfo)
            ):
                pass_list.append(r["pass_type"])

        if not pass_list:
            raise ValueError(
                f"Quantiation for precision {precision}, algorithm {algo} "
                f"and implementation {impl} is not supported"
            )
        logger.info("pass list: %s", pass_list)
        return pass_list

    def _get_passes_dict(self, pass_list):
        precision_in_bits = self._get_precision_bits()
        wtypes = self._get_precision_in_wtypes()
        quant_format = "QOperator"
        if self.args.use_qdq_encoding:
            quant_format = "QDQ"

        # config options to add for a given option
        to_add = {
            "AutoAWQQuantizer": {"w_bits": precision_in_bits},
            "GptqQuantizer": {"bits": precision_in_bits},
            "OnnxBnB4Quantization": {"quant_type": precision_in_bits},
            "NVModelOptQuantization": {"precision": precision_in_bits, "algorithm": self.args.algorithm.upper()},
            "OnnxDynamicQuantization": {"weight_type": wtypes, "quant_format": quant_format},
            "OnnxStaticQuantization": {
                "weight_type": wtypes,
                "quant_format": quant_format,
                "data_config": "default_data_config",
            },
            "OnnxMatMul4Quantizer": {"quant_format": quant_format},
            "IncDynamicQuantization": {"algorithm": self.args.algorithm.upper(), "bits": precision_in_bits},
        }

        passes_dict = {}
        for p in pass_list:
            pd = {"type": p}
            if to_add.get(p) is not None:
                pd.update(dict(to_add[p].items()))
            passes_dict[p.lower()] = pd
        logger.info("selected pass configs: %s", passes_dict)
        return passes_dict

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
        plist = self._get_pass_list(self.args.precision, self.args.algorithm, self.args.implementation, is_hf_model)

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
    "output_dir": "models",
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}

# Pass order in this mapping is important. More than one passes could be selected from this mapping.
PT_QUANT_IMPLEMENTATION_MAPPING = [
    {"impl_name": "quarot", "pass_type": "QuaRot"},
    {"impl_name": "spinquant", "pass_type": "SpinQuant"},
    {"impl_name": "awq", "pass_type": "AutoAWQQuantizer"},
    {"impl_name": "autogptq", "pass_type": "GptqQuantizer"},
]

# Pass order in this mapping is important. More than one passes could be selected from this mapping.
ONNX_QUANT_IMPLEMENTATION_MAPPING = [
    {"impl_name": "bnb", "pass_type": "OnnxBnB4Quantization"},
    {"impl_name": "ort", "pass_type": "OnnxMatMul4Quantizer"},
    {"impl_name": "ort", "pass_type": "OnnxDynamicQuantization"},
    {"impl_name": "ort", "pass_type": "OnnxStaticQuantization"},
    {"impl_name": "nvmo", "pass_type": "NVModelOptQuantization"},
    {"impl_name": "inc", "pass_type": "IncDynamicQuantization"},
    # "impl_name": "inc", "pass_type": "IncStaticQuantization"},
]
