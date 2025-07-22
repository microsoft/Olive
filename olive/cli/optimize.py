# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# ruff: noqa: T201

from argparse import ArgumentParser
from collections import OrderedDict
from copy import deepcopy
from typing import Any

from olive.cli.base import (
    BaseOliveCLICommand,
    add_input_model_options,
    add_logging_options,
    add_save_config_file_options,
    get_input_model_config,
)
from olive.common.utils import set_nested_dict_value
from olive.constants import Precision
from olive.hardware.constants import ExecutionProvider


class OptimizeCommand(BaseOliveCLICommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        sub_parser = parser.add_parser(
            "optimize",
            help="Optimize the input model with comprehensive pass scheduling",
        )

        # Model options
        add_input_model_options(
            sub_parser,
            enable_hf=True,
            enable_hf_adapter=True,
            enable_pt=True,
            enable_onnx=True,
            default_output_path="optimized-model",
        )

        # Execution provider options
        sub_parser.add_argument(
            "--provider",
            type=str,
            default=ExecutionProvider.CPUExecutionProvider,
            choices=[ep.value for ep in ExecutionProvider],
            help="Execution provider (EP) to use for optimization.",
        )

        # Device options
        sub_parser.add_argument(
            "--device",
            type=str,
            default="cpu",
            choices=["cpu", "gpu", "npu"],
            help="Target device for optimization.",
        )

        # Precision options
        sub_parser.add_argument(
            "--precision",
            type=str,
            default=Precision.FP32,
            choices=[p.value for p in Precision],
            help="Target precision for optimization.",
        )

        # Optional activation precision
        sub_parser.add_argument(
            "--act_precision",
            type=str,
            choices=[p.value for p in Precision],
            help="Activation precision for quantization (optional).",
        )

        # Model splitting options
        sub_parser.add_argument(
            "--num_split",
            type=int,
            help="Number of splits for model splitting (optional).",
        )

        sub_parser.add_argument(
            "--memory",
            type=int,
            help="Available device memory in MB (optional).",
        )

        # Exporter options
        sub_parser.add_argument(
            "--exporter",
            type=str,
            choices=["model_builder", "dynamo_exporter", "torchscript_exporter", "optimum_exporter"],
            help="Exporter to use for model conversion (optional).",
        )

        # Dynamic shape options
        sub_parser.add_argument(
            "--dim_param",
            type=str,
            nargs="*",
            help="Dynamic parameter names for dynamic to fixed shape conversion (optional).",
        )

        sub_parser.add_argument(
            "--dim_value",
            type=int,
            nargs="*",
            help="Fixed dimension values for dynamic to fixed shape conversion (optional).",
        )

        # QDQ format option
        sub_parser.add_argument(
            "--use_qdq_format",
            action="store_true",
            help="Use QDQ format for quantization instead of QOperator format.",
        )

        # Graph surgeries option
        sub_parser.add_argument(
            "--surgeries",
            type=str,
            nargs="*",
            help="List of graph surgeries to apply (optional).",
        )

        add_logging_options(sub_parser)
        add_save_config_file_options(sub_parser)
        sub_parser.set_defaults(func=OptimizeCommand)

    def run(self):
        return self._run_workflow()

    def _get_run_config(self, tempdir: str) -> dict[str, Any]:
        config = deepcopy(TEMPLATE)

        # Set input model configuration
        config["input_model"] = get_input_model_config(self.args)
        is_hf_model = config["input_model"]["type"].lower() == "hfmodel"

        # Validate device and provider compatibility
        self._validate_device_provider_compatibility()

        # Build the pass list based on conditions
        passes_config = self._build_passes_config(is_hf_model)
        config["passes"] = passes_config

        # Set system configuration
        self._update_system_config(config)

        # Apply customizations
        to_replace = [
            ("output_dir", self.args.output_path),
            ("log_severity_level", self.args.log_level),
        ]
        for keys, value in to_replace:
            if value is not None:
                set_nested_dict_value(config, keys, value)

        return config

    def _validate_device_provider_compatibility(self):
        """Validate that device and provider are compatible."""
        provider = ExecutionProvider(self.args.provider)

        # Auto-adjust device based on provider if needed
        if provider == ExecutionProvider.DmlExecutionProvider and self.args.device not in ["gpu", "npu"]:
            self.args.device = "gpu"
        elif provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]:
            self.args.device = "npu"
        elif provider == ExecutionProvider.CUDAExecutionProvider:
            self.args.device = "gpu"

    def _update_system_config(self, config: dict[str, Any]):
        """Update system configuration based on provider and device."""
        provider = ExecutionProvider(self.args.provider)

        config["systems"]["local_system"]["accelerators"] = [
            {"device": self.args.device, "execution_providers": [provider.value]}
        ]

    def _build_passes_config(self, is_hf_model: bool) -> dict[str, Any]:
        """Build the passes configuration based on user selections and conditions."""
        passes_config = OrderedDict()

        provider = ExecutionProvider(self.args.provider)
        precision = Precision(self.args.precision)

        # Helper function to check if precision is quantized
        def is_quantized_precision(p):
            return p in [Precision.INT4, Precision.INT8, Precision.UINT4, Precision.UINT8]

        # Schedule passes in the specified order

        # 1. QuaRot
        if (
            is_quantized_precision(precision)
            and is_hf_model
            and provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]
        ):
            passes_config["quarot"] = {"type": "QuaRot"}

        # 2. Gptq
        if (
            is_hf_model
            and is_quantized_precision(precision)
            and provider != ExecutionProvider.OpenVINOExecutionProvider
        ):
            passes_config["gptq"] = {"type": "Gptq", "bits": self._precision_to_bits(precision)}

        # 3. CaptureSplitInfo
        if is_hf_model and (self.args.num_split is not None or self.args.memory is not None):
            passes_config["capture_split_info"] = {"type": "CaptureSplitInfo"}
            passes_config["capture_split_info"]["unique_embeds_lm_head_splits"] = True
            if self.args.num_split is not None:
                passes_config["capture_split_info"]["num_splits"] = self.args.num_split
            if self.args.memory is not None:
                passes_config["capture_split_info"]["memory"] = self.args.memory

        # 4. ModelBuilder
        if (
            is_hf_model
            and provider != ExecutionProvider.OpenVINOExecutionProvider
            and self.args.exporter == "model_builder"
        ):
            passes_config["model_builder"] = {"type": "ModelBuilder", "precision": precision.value}
            if precision.value == Precision.INT4:
                passes_config["model_builder"]["int4_block_size"] = 32
                passes_config["model_builder"]["int4_accuracy_level"] = 4
                passes_config["model_builder"]["int4_op_types_to_quantize"] = ["MatMul", "Gather"]

        # 5. OnnxConversion
        if (
            is_hf_model
            and provider != ExecutionProvider.OpenVINOExecutionProvider
            and self.args.exporter in ["dynamo_exporter", "torchscript_exporter"]
        ):
            passes_config["onnx_conversion"] = {
                "type": "OnnxConversion",
                "use_dynamo_exporter": self.args.exporter == "dynamo_exporter",
                "torch_dtype": "float32",
            }

        # 6. OptimumConversion
        if (
            is_hf_model
            and provider != ExecutionProvider.OpenVINOExecutionProvider
            and self.args.exporter == "optimum_exporter"
        ):
            passes_config["optimum_conversion"] = {"type": "OptimumConversion"}

        # 7. OptimumOpenvinoConversion
        if is_hf_model and provider == ExecutionProvider.OpenVINOExecutionProvider:
            passes_config["optimum_openvino_conversion"] = {"type": "OpenVINOConversion"}

        # 8. DynamicToFixedShape
        if (
            provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]
            and self.args.dim_param is not None
            and self.args.dim_value is not None
        ):
            passes_config["dynamic_to_fixed_shape"] = {
                "type": "DynamicToFixedShape",
                "dim_param": self.args.dim_param,
                "dim_value": self.args.dim_value,
            }

        # 9. InputNCHWtoNHWC (For VitisAI, this would be part of VitisAIQuantization)
        if provider == ExecutionProvider.VitisAIExecutionProvider:
            # VitisAI preprocessing pass
            passes_config["vitis_ai_preprocess"] = {"type": "VitisAIQuantization"}

        # 10. OpenVINOIoUpdate
        if provider == ExecutionProvider.OpenVINOExecutionProvider and is_hf_model:
            passes_config["openvino_io_update"] = {"type": "OpenVINOConversion"}

        # 11. OnnxPeepholeOptimizer
        if self.args.exporter != "model_builder":
            passes_config["onnx_peephole_optimizer"] = {"type": "OnnxPeepholeOptimizer"}

        # 12. MatMulNBitsToQDQ
        if is_hf_model and "gptq" in passes_config and self.args.use_qdq_format:
            passes_config["matmul_nbits_to_qdq"] = {"type": "MatMulNBitsToQDQ"}

        # 13. GraphSurgeries
        if self.args.surgeries is not None:
            surgeries_list = [{"surgeon": item} for item in self.args.surgeries[0].split(",")]
            passes_config["graph_surgeries"] = {"type": "GraphSurgeries", "surgeries": surgeries_list}

        # 14. OnnxBlockWiseRtnQuantization
        if not is_hf_model and precision == Precision.INT4:
            passes_config["onnx_blockwise_rtn_quantization"] = {"type": "OnnxBlockWiseRtnQuantization"}

        # 15. OnnxFloatToFloat16
        if precision == Precision.FP16:
            passes_config["onnx_float_to_float16"] = {"type": "OnnxFloatToFloat16"}

        # 16. OnnxStaticQuantization
        act_precision_check = (
            self.args.act_precision
            in [Precision.INT8.value, Precision.UINT8.value, Precision.INT16.value, Precision.UINT16.value]
            if self.args.act_precision
            else False
        )
        precision_check = (
            precision in [Precision.INT8, Precision.UINT8, Precision.INT16, Precision.UINT16]
            and "gptq" not in passes_config
        )

        if precision_check or act_precision_check:
            passes_config["onnx_static_quantization"] = {
                "type": "OnnxStaticQuantization",
                "precision": precision.value,
                "act_precision": self.args.act_precision,
                "quant_format": "QDQ" if self.args.use_qdq_format else "QOperator",
            }

        # 17. OrtTransformersOptimization
        if self.args.exporter in ["torchscript_exporter", "dynamo_exporter"]:
            passes_config["ort_transformers_optimization"] = {
                "type": "OrtTransformersOptimization",
                "opt_level": 0,
                "float16": precision == Precision.FP16,
            }

        # 18. SplitModel
        if is_hf_model and (self.args.num_split is not None or self.args.memory is not None):
            passes_config["split_model"] = {"type": "SplitModel"}

        # 19. StaticLLM
        if provider in [ExecutionProvider.QNNExecutionProvider, ExecutionProvider.VitisAIExecutionProvider]:
            passes_config["static_llm"] = {"type": "StaticLLM"}

        # 20. VitisAIAddMetaData
        if provider == ExecutionProvider.VitisAIExecutionProvider:
            passes_config["vitis_ai_add_metadata"] = {"type": "VitisAIAddMetaData"}

        # 21. EPContextBinaryGenerator
        if provider == ExecutionProvider.QNNExecutionProvider:
            passes_config["ep_context_binary_generator"] = {"type": "QNNContextBinaryGenerator"}

        # 22. ComposeOnnxModels
        if (
            is_hf_model
            and (self.args.num_split is not None or self.args.memory is not None)
            and provider == ExecutionProvider.QNNExecutionProvider
        ):
            passes_config["compose_onnx_models"] = {"type": "ComposeOnnxModels"}

        # 23. OpenVINOEncapsulation
        if is_hf_model and provider == ExecutionProvider.OpenVINOExecutionProvider:
            passes_config["openvino_encapsulation"] = {"type": "OpenVINOEncapsulation"}

        return passes_config

    def _precision_to_bits(self, precision: Precision) -> int:
        """Convert precision enum to bit count."""
        precision_bits_map = {
            Precision.INT4: 4,
            Precision.UINT4: 4,
            Precision.INT8: 8,
            Precision.UINT8: 8,
            Precision.INT16: 16,
            Precision.UINT16: 16,
            Precision.INT32: 32,
            Precision.UINT32: 32,
        }
        return precision_bits_map.get(precision, 32)


# Template configuration for the optimize command
TEMPLATE = {
    "input_model": {"type": "HfModel"},
    "systems": {
        "local_system": {
            "type": "LocalSystem",
            "accelerators": [{"device": "cpu", "execution_providers": ["CPUExecutionProvider"]}],
        }
    },
    "passes": OrderedDict(),
    "host": "local_system",
    "target": "local_system",
    "no_artifacts": True,
}
