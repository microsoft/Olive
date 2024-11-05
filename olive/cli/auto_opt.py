# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from typing import Any, Dict, List

from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_search_options,
    add_shared_cache_options,
    get_input_model_config,
    is_remote_run,
    save_output_model,
    update_accelerator_options,
    update_remote_options,
    update_search_options,
    update_shared_cache_options,
)
from olive.common.utils import hardlink_copy_dir, set_nested_dict_value
from olive.package_config import OlivePackageConfig


class AutoOptCommand(BaseOliveCLICommand):

    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "auto-opt",
            help="Automatically optimize the performance of the input model.",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="auto-opt-output",
        )

        # add accelerator options
        add_accelerator_options(sub_parser)

        # dataset options
        sub_parser.add_argument(
            "-d",
            "--data_name",
            type=str,
            help="The dataset name.",
        )
        sub_parser.add_argument(
            "--split",
            type=str,
            help="The dataset split to use for evaluation.",
        )
        sub_parser.add_argument(
            "--subset",
            type=str,
            help="The dataset subset to use for evaluation.",
        )
        sub_parser.add_argument(
            "--input_cols",
            type=str,
            nargs="*",
            help="The input columns to use for evaluation.",
        )
        sub_parser.add_argument(
            "--batch_size",
            type=int,
            default=1,
            help="Batch size for evaluation.",
        )

        sub_parser.add_argument(
            "--precision",
            type=str,
            default="fp32",
            choices=["fp4", "fp8", "fp16", "fp32", "int4", "int8", "int16", "int32", "nf4"],
            help=(
                "The output precision of the optimized model. If not specified, "
                "the default precision is fp32 for cpu and fp16 for gpu"
            ),
        )
        sub_parser.add_argument(
            "--use_model_builder",
            action="store_true",
            help=(
                "Whether to use model builder pass for optimization, enable only "
                "when the model is supported by model builder"
            ),
        )

        # DynamicToFixedShape options
        sub_parser.add_argument(
            "--dynamic-to-fixed-shape-dim-param",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Symbolic parameter names to use for dynamic to fixed shape pass. "
                "Required only when DynamicToFixedShape pass is enabled."
            ),
        )
        sub_parser.add_argument(
            "--dynamic-to-fixed-shape-dim-value",
            type=int,
            nargs="*",
            default=None,
            help=(
                "Symbolic parameter values to use for dynamic to fixed shape pass. "
                "Required only when DynamicToFixedShape pass is enabled."
            ),
        )

        # Split options
        split_group = sub_parser.add_mutually_exclusive_group(required=False)
        split_group.add_argument(
            "--num-splits",
            type=int,
            help="Number of splits to use for model splitting. Input model must be an HfModel.",
        )
        split_group.add_argument(
            "--cost-model",
            type=str,
            help="Path to the cost model file to use for model splitting. Mutually exclusive with num-splits.",
        )
        # TODO(jambayk): Move this to device options
        sub_parser.add_argument("--device-memory", type=int, help="Device memory in bytes to use for model splitting.")

        # MixedPrecisionOverrides options
        sub_parser.add_argument(
            "--mixed-precision-overrides-config",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Dictionary of name to precision. Has to be even number of entreis with even "
                "entries being the keys and odd entries being the values. "
                'Required only when output precision is "fp16" and MixedPrecisionOverrides pass is enabled.'
            ),
        )

        add_search_options(sub_parser)
        add_remote_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        sub_parser.set_defaults(func=AutoOptCommand)

    def run(self):
        from olive.workflows import run as olive_run

        with tempfile.TemporaryDirectory(prefix="olive-cli-tmp-", dir=self.args.output_path) as tempdir:
            run_config = self.get_run_config(tempdir)
            olive_run(run_config)

            if is_remote_run(self.args):
                return

            if run_config.get("search_strategy"):
                # TODO(anyone): maybe save the best model instead of just the search results
                hardlink_copy_dir(run_config["output_dir"], self.args.output_path)
                print(f"Search results are saved to {self.args.output_path}")
            else:
                save_output_model(run_config, self.args.output_path)

    def get_run_config(self, tempdir) -> Dict:
        config = deepcopy(TEMPLATE)
        olive_config = OlivePackageConfig.load_default_config()

        if (self.args.provider == "DmlExecutionProvider") and (self.args.device not in ["gpu", "npu"]):
            # Force the device to gpu for Direct ML provider
            self.args.device = "gpu"

        to_replace = [
            ("input_model", get_input_model_config(self.args)),
            ("output_dir", tempdir),
            ("log_severity_level", self.args.log_level),
        ]
        if self.args.enable_search is None:
            to_replace.append(("passes", self._get_passes_config(config, olive_config)))
        elif self.args.enable_search:
            excluded_passes = [
                "OrtPerfTuning",
                "OnnxConversion" if self.args.use_model_builder else "ModelBuilder",
            ]
            to_replace.extend(
                [
                    ("data_configs", self._get_data_config()),
                    (
                        "auto_optimizer_config",
                        {"precisions": [self.args.precision], "excluded_passes": excluded_passes},
                    ),
                ]
            )

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        update_accelerator_options(self.args, config)
        update_search_options(self.args, config)
        update_remote_options(config, self.args, "auto-opt", tempdir)
        update_shared_cache_options(config, self.args)

        if self.args.enable_search is None:
            del config["evaluators"]
            del config["evaluator"]
            del config["search_strategy"]
            del config["auto_optimizer_config"]
        elif self.args.enable_search:
            del config["passes"]
            if not config["data_configs"]:
                raise ValueError("Dataset is required when search is enabled")

        return config

    def _get_data_config(self) -> List[Dict[str, Any]]:
        if not self.args.data_name:
            return []

        to_replace = [
            (("load_dataset_config", "data_name"), self.args.data_name),
            (("load_dataset_config", "split"), self.args.split),
            (("load_dataset_config", "subset"), self.args.subset),
            (("pre_process_data_config", "input_cols"), self.args.input_cols),
            (("dataloader_config", "batch_size"), self.args.batch_size),
        ]
        data_config = {
            "name": "data_config",
            "type": "HuggingfaceContainer",
            "load_dataset_config": {},
            "pre_process_data_config": {},
            "dataloader_config": {},
        }
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(data_config, keys, value)

        return [data_config]

    def _get_passes_config(self, config: Dict[str, Any], olive_config: OlivePackageConfig) -> Dict[str, Any]:
        if self.args.mixed_precision_overrides_config and len(self.args.mixed_precision_overrides_config) % 2 != 0:
            raise ValueError("Even number of entries required for mixed precision overrides config.")

        passes_config: Dict[str, Any] = config["passes"]
        mixed_precision_overrides_config = (
            {
                self.args.mixed_precision_overrides_config[i]: self.args.mixed_precision_overrides_config[i + 1]
                for i in range(0, len(self.args.mixed_precision_overrides_config), 2)
            }
            if self.args.mixed_precision_overrides_config
            else None
        )

        to_replace = [
            (("capture_split_info", "num_splits"), self.args.num_splits),
            (("capture_split_info", "cost_model"), self.args.cost_model),
            (("capture_split_info", "max_memory"), self.args.device_memory),
            (("bnb4", "quant_type"), PRECISION_MAPPING["bnb4"].get(self.args.precision, self.args.precision)),
            (
                ("dynamic_quant", "weight_type"),
                PRECISION_MAPPING["dynamic_quant"].get(self.args.precision, self.args.precision),
            ),
            (
                ("model_builder", "precision"),
                PRECISION_MAPPING["model_builder"].get(self.args.precision, self.args.precision),
            ),
            (("model_builder", "metadata_only"), config["input_model"]["type"].lower() == "onnxmodel"),
            (("transformer_optimizer", "use_fp16"), self.args.precision == "fp16"),
            (("to_fixed_shape", "dim_param"), self.args.dynamic_to_fixed_shape_dim_param),
            (("to_fixed_shape", "dim_value"), self.args.dynamic_to_fixed_shape_dim_value),
            (("mixed_precision_overrides", "overrides_config"), mixed_precision_overrides_config),
        ]
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(passes_config, keys, value)

        if self.args.num_splits is None and self.args.cost_model is None:
            del passes_config["capture_split_info"], passes_config["split_model"]
        if self.args.cost_model is not None and self.args.device_memory is None:
            raise ValueError("device_memory is required if cost_model is provided.")

        del passes_config["conversion" if self.args.use_model_builder else "model_builder"]
        # Remove dynamic-to-fixed-shape pass if not required
        if (self.args.dynamic_to_fixed_shape_dim_param is None) ^ (self.args.dynamic_to_fixed_shape_dim_value is None):
            raise ValueError(
                "Only one of dynamic-to-fixed-shape-dim-param and dynamic-to-fixed-shape-dim-value is set. Provide both"
                " or none."
            )
        elif self.args.dynamic_to_fixed_shape_dim_param is None:
            del passes_config["to_fixed_shape"]
        # Remove mixed_precision_overrides pass if not required
        if mixed_precision_overrides_config is None:
            del passes_config["mixed_precision_overrides"]

        for pass_name in list(passes_config.keys()):
            pass_run_config = passes_config[pass_name]
            pass_module_config = olive_config.get_pass_module_config(pass_run_config["type"])
            if (
                (self.args.precision not in pass_module_config.supported_precisions)
                or (self.args.provider not in pass_module_config.supported_providers)
                or (self.args.device not in pass_module_config.supported_accelerators)
            ):
                del passes_config[pass_name]

        if ("conversion" not in passes_config) and ("model_builder" not in passes_config):
            raise ValueError("Cannot export an onnx model with combination of provided options.")

        return passes_config


EVALUATE_TEMPLATE = {
    "common_evaluator": {
        "metrics": [
            {
                "name": "accuracy",
                "type": "accuracy",
                "sub_types": [
                    {"name": "accuracy_score", "priority": 1, "goal": {"type": "max-degradation", "value": 0.1}},
                ],
                "data_config": "data_config",
            },
            {
                "name": "latency",
                "type": "latency",
                "sub_types": [
                    {"name": "avg", "priority": 2, "goal": {"type": "percent-min-improvement", "value": 1}},
                ],
                "data_config": "data_config",
                "user_config": {"io_bind": True},
            },
        ]
    }
}

TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "auto_optimizer_config": {},
    "search_strategy": {"execution_order": "joint", "search_algorithm": "tpe", "num_samples": 5, "seed": 0},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": OrderedDict(
        [
            ("capture_split_info", {"type": "CaptureSplitInfo"}),
            ("conversion", {"type": "OnnxConversion"}),
            ("model_builder", {"type": "ModelBuilder", "precision": "fp32", "metadata_only": False}),
            (
                "transformer_optimizer",
                {"type": "OrtTransformersOptimization", "opt_level": 0, "use_fp16": False, "keep_io_types": False},
            ),
            ("optimizer", {"type": "OnnxModelOptimizer"}),
            ("fp16_to_fp32", {"type": "OnnxIOFloat16ToFloat32"}),
            ("qnn_preprocess", {"type": "QNNPreprocess"}),
            ("mixed_precision_overrides", {"type": "MixedPrecisionOverrides", "overrides_config": None}),
            ("dynamic_quant", {"type": "OnnxQuantization", "weight_type": "QInt8"}),
            ("matmul4", {"type": "OnnxMatMul4Quantizer"}),
            ("bnb4", {"type": "OnnxBnb4Quantization", "quant_type": "nf4"}),
            ("to_fixed_shape", {"type": "DynamicToFixedShape", "dim_param": None, "dim_value": None}),
            ("mnb_to_qdq", {"type": "MatMulNBitsToQDQ"}),
            ("split_model", {"type": "SplitModel"}),
            ("extract_adapters", {"type": "ExtractAdapters"}),
        ]
    ),
    "host": "local_system",
    "evaluators": EVALUATE_TEMPLATE,
    "evaluator": "common_evaluator",
    "target": "local_system",
}

PRECISION_MAPPING = {
    "capture_split_info": {},
    "conversion": {},
    "model_builder": {},
    "transformer_optimizer": {},
    "optimizer": {},
    "fp16_to_fp32": {},
    "qnn_preprocess": {},
    "dynamic_quant": {"int8": "QInt8", "uint8": "QUInt8"},
    "matmul4": {},
    "bnb4": {},
    "to_fixed_shape": {},
    "mixed_precision_overrides": {},
    "mixed_precision": {},
    "mnb_to_qdq": {},
    "split_model": {},
    "extract_adapters": {},
}
