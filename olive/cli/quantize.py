# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201
# ruff: noqa: RUF012

from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

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
from olive.constants import Precision, PrecisionBits, QuantAlgorithm
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
            choices=[a.value for a in QuantAlgorithm],
            help="List of quantization algorithms to run.",
        )
        sub_parser.add_argument(
            "--precision",
            type=str,
            default="int8",
            choices=list(Precision) + list(PrecisionBits),
            help="The precision of the quantized model.",
        )
        sub_parser.add_argument(
            "--implementation",
            type=str,
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
            ):
                if not self._check_data_name_arg(pinfo):
                    print(
                        f"Warning: Quantization for {algo} {precision} {impl} implementation with QDQ {self.args.use_qdq_encoding}"
                        " requires dataset. Please provide a dataset using --data_name option."
                    )
                else:
                    pass_list.append(r["pass_type"])

        if not pass_list:
            raise ValueError(
                f"Quantiation for precision {precision}, algorithm {algo} and implementation {impl} "
                f"with QDQ {self.args.use_qdq_encoding} is not supported"
            )
        print(f"pass list: {pass_list}")
        return pass_list

    def _get_passes_dict(self, pass_list):
        quant_format = "QDQ" if self.args.use_qdq_encoding else "QOperator"

        # config options to add for a given option
        to_add = {
            "AutoAWQQuantizer": {"bits": self.args.precision},
            "GptqQuantizer": {"bits": self.args.precision},
            "OnnxBnB4Quantization": {"precision": self.args.precision},
            "NVModelOptQuantization": {"precision": self.args.precision, "algorithm": self.args.algorithm},
            "OnnxDynamicQuantization": {"precision": self.args.precision, "quant_format": quant_format},
            "OnnxStaticQuantization": {
                "precision": self.args.precision,
                "quant_format": quant_format,
                "data_config": "default_data_config",
            },
            "OnnxMatMul4Quantizer": {"quant_format": quant_format},
            "IncDynamicQuantization": {"algorithm": self.args.algorithm, "bits": self.args.precision},
        }

        passes_dict = {}
        for p in pass_list:
            pd = {"type": p}
            if to_add.get(p) is not None:
                pd.update(dict(to_add[p].items()))
            passes_dict[p.lower()] = pd
        print(f"selected pass configs: {passes_dict}")
        return passes_dict

    def _customize_config(self, config):
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for k, v in to_replace:
            if v is not None:
                set_nested_dict_value(config, k, v)

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
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
    {"pass_type": "OnnxHqqQuantization"},
    # "impl_name": "inc", "pass_type": "IncStaticQuantization"},
]
