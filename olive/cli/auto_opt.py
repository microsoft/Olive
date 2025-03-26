# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from argparse import ArgumentParser
from copy import deepcopy
from typing import Any

from olive.auto_optimizer import AutoOptimizer, AutoOptimizerConfig
from olive.cli.base import (
    BaseOliveCLICommand,
    add_accelerator_options,
    add_input_model_options,
    add_logging_options,
    add_remote_options,
    add_save_config_file_options,
    add_search_options,
    add_shared_cache_options,
    get_input_model_config,
    update_accelerator_options,
    update_remote_options,
    update_search_options,
    update_shared_cache_options,
)
from olive.common.utils import set_nested_dict_value
from olive.constants import Precision
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ModelConfig


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
            type=Precision,
            default=Precision.FP32,
            choices=[v.value for v in Precision],
            help=(
                "The output precision of the optimized model. If not specified, "
                "the default precision is fp32 for cpu and fp16 for gpu"
            ),
        )
        sub_parser.add_argument(
            "--use_dynamo_exporter",
            action="store_true",
            help="Whether to use dynamo_export API to export ONNX model.",
        )
        sub_parser.add_argument(
            "--use_model_builder",
            action="store_true",
            help=(
                "Whether to use model builder pass for optimization, enable only "
                "when the model is supported by model builder"
            ),
        )
        sub_parser.add_argument(
            "--use_qdq_encoding",
            action="store_true",
            help=(
                "Whether to use QDQ encoding for quantized operators instead of ONNXRuntime contrib operators like"
                " MatMulNBits"
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
                "Required only when using QNNExecutionProvider."
            ),
        )
        sub_parser.add_argument(
            "--dynamic-to-fixed-shape-dim-value",
            type=int,
            nargs="*",
            default=None,
            help=(
                "Symbolic parameter values to use for dynamic to fixed shape pass. "
                "Required only when using QNNExecutionProvider."
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
            help=(
                "Path to the cost model csv file to use for model splitting. Mutually exclusive with num-splits. Must"
                " be a csv with headers `module,num_params,num_bytes,num_flops` where each row corresponds to the name"
                " or a module (with no children), the number of parameters, the number of bytes, and the number of"
                " FLOPs(batch_size=1, seqlen=1) the module uses when in the desired precision."
            ),
        )

        # MixedPrecisionOverrides options
        sub_parser.add_argument(
            "--mixed-precision-overrides-config",
            type=str,
            nargs="*",
            default=None,
            help=(
                "Dictionary of name to precision. Has to be even number of entries with even "
                "entries being the keys and odd entries being the values. "
                'Required only when output precision is "fp16" and MixedPrecisionOverrides pass is enabled.'
            ),
        )

        sub_parser.add_argument(
            "--use_ort_genai", action="store_true", help="Use OnnxRuntime generate() API to run the model"
        )

        sub_parser.add_argument(
            "--surgeries", type=str, nargs="*", default=None, help="List of graph surgeries to apply."
        )

        add_search_options(sub_parser)
        add_remote_options(sub_parser)
        add_shared_cache_options(sub_parser)
        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        sub_parser.set_defaults(func=AutoOptCommand)

    def run(self):
        self._run_workflow()

    @property
    def _auto_optimizer_config(self):
        return AutoOptimizerConfig.parse_obj(
            {
                "precision": self.args.precision,
                "accelerator": AcceleratorSpec(
                    accelerator_type=self.args.device,
                    execution_provider=self.args.provider,
                    memory=self.args.memory,
                ),
                "finetune": False,
                "eval_data_config": None,
                "train_data_config": None,
                "calibration_data_config": None,
                "convert_to_onnx": True,
                "use_model_builder": self.args.use_model_builder,
                "use_dynamo_exporter": self.args.use_dynamo_exporter,
                "quantize": True,
                "generate_config_only": False,
            }
        )

    def _get_run_config(self, tempdir) -> dict:
        config = deepcopy(TEMPLATE)

        # TODO(anyone): Change add_accelerator_options to have no default device, this can be inferred
        # by create_accelerators
        if (self.args.provider == "DmlExecutionProvider") and (self.args.device not in ["gpu", "npu"]):
            # Force the device to gpu for Direct ML provider
            self.args.device = "gpu"
        elif self.args.provider in ["QNNExecutionProvider", "VitisAIExecutionProvider"]:
            self.args.device = "npu"
        elif self.args.provider == "CUDAExecutionProvider":
            self.args.device = "gpu"

        # _get_passes_config requires input_model to be set
        config["input_model"] = get_input_model_config(self.args)

        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
            ("passes", self._get_passes_config(config)),
        ]

        if self.args.enable_search:
            to_replace.extend(
                [
                    ("data_configs", self._get_data_config()),
                    ("auto_optimizer_config", self._auto_optimizer_config),
                ]
            )

        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        update_accelerator_options(self.args, config)
        update_search_options(self.args, config)
        update_remote_options(config, self.args, "auto-opt", tempdir)
        update_shared_cache_options(config, self.args)

        if self.args.enable_search:
            if not config["data_configs"]:
                raise ValueError("Dataset is required when search is enabled")
        else:
            del config["data_configs"]
            del config["evaluators"]
            del config["evaluator"]
            del config["search_strategy"]
            del config["auto_optimizer_config"]

        return config

    def _get_data_config(self) -> list[dict[str, Any]]:
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

    def _get_passes_config(self, config: dict[str, Any]) -> dict[str, Any]:
        if self.args.mixed_precision_overrides_config and len(self.args.mixed_precision_overrides_config) % 2 != 0:
            raise ValueError("Even number of entries required for mixed precision overrides config.")

        if self.args.cost_model is not None and self.args.memory is None:
            raise ValueError("memory is required if cost_model is provided.")

        if self.args.provider == "QNNExecutionProvider" and not (
            self.args.dynamic_to_fixed_shape_dim_param and self.args.dynamic_to_fixed_shape_dim_value
        ):
            raise ValueError(
                "dynamic-to-fixed-shape-dim-param and dynamic-to-fixed-shape-dim-value are required "
                "when using QNNExecutionProvider."
            )

        mixed_precision_overrides_config = (
            {
                self.args.mixed_precision_overrides_config[i]: self.args.mixed_precision_overrides_config[i + 1]
                for i in range(0, len(self.args.mixed_precision_overrides_config), 2)
            }
            if self.args.mixed_precision_overrides_config
            else None
        )

        surgeries = [{"surgeon": surgeon} for surgeon in self.args.surgeries] if self.args.surgeries else None

        to_replace = [
            (("capture_split_info", "num_splits"), self.args.num_splits),
            (("capture_split_info", "cost_model"), self.args.cost_model),
            (("conversion", "save_metadata_for_token_generation"), self.args.use_ort_genai),
            (("model_builder", "metadata_only"), self.args.use_model_builder or not self.args.use_ort_genai),
            # select the float dtype based on the precision, int4 only quantizes matmuls
            # so we still need to set the float precision separately
            (
                ("transformer_optimizer", "float16"),
                self.args.precision == Precision.FP16
                or (self.args.precision == Precision.INT4 and self.args.provider != "CPUExecutionProvider"),
            ),
            (("to_fixed_shape", "dim_param"), self.args.dynamic_to_fixed_shape_dim_param),
            (("to_fixed_shape", "dim_value"), self.args.dynamic_to_fixed_shape_dim_value),
            (("mixed_precision_overrides", "overrides_config"), mixed_precision_overrides_config),
            (("surgeries", "surgeries"), surgeries),
        ]

        passes_configs: dict[str, Any] = config["passes"]
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(passes_configs, keys, value)

        auto_optimizer = AutoOptimizer(
            model_config=ModelConfig.parse_obj(config["input_model"]),
            optimizer_config=self._auto_optimizer_config,
        )
        passes_configs = auto_optimizer.generate_run_passes_configs(passes_configs)
        passes_configs = {
            name: [rpc.dict() for rpc in run_pass_configs] for name, run_pass_configs in passes_configs.items()
        }

        # check that there is at least one capture pass for non-onnx models
        if (config["input_model"]["type"].lower() != "onnxmodel") and ("onnx_conversion" not in passes_configs):
            raise ValueError("Cannot export an onnx model with combination of provided options.")

        return passes_configs


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
    "auto_optimizer_config": None,
    "data_configs": {},
    "search_strategy": {"execution_order": "joint", "sampler": "tpe", "max_samples": 5, "seed": 0},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": {
        # pytorch related passes
        "capture_split_info": {"type": "CaptureSplitInfo"},
        "conversion": {"type": "OnnxConversion", "torch_dtype": "float32", "save_metadata_for_token_generation": False},
        "model_builder": {"type": "ModelBuilder", "precision": Precision.FP32, "metadata_only": False},
        # use transformer optimizer for fp16 conversion too
        # opt_level set to 0 to avoid graph transformations done by onnxruntime inference sessions
        # that are incompatible with later passes. opt_level > 0 is optional and can be done during session creation
        "transformer_optimizer": {
            "type": "OrtTransformersOptimization",
            "opt_level": 0,
            "float16": False,
            "keep_io_types": False,
        },
        # qnn preparation passes
        "to_fixed_shape": {"type": "DynamicToFixedShape", "dim_param": None, "dim_value": None},
        "mixed_precision_overrides": {"type": "MixedPrecisionOverrides", "overrides_config": None},
        # post processing passes
        "surgeries": {"type": "GraphSurgeries", "surgeries": []},
    },
    "host": "local_system",
    "evaluators": EVALUATE_TEMPLATE,
    "evaluator": "common_evaluator",
    "target": "local_system",
    "no_artifacts": True,
}
