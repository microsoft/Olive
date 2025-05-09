# -----------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# -----------------------------------------------------------------------------

import logging
from copy import deepcopy
from typing import Any, ClassVar, Optional, Union

from olive.common.config_utils import ConfigBase
from olive.common.pydantic_v1 import Field
from olive.constants import Precision, PrecisionBits, QuantAlgorithm, QuantEncoding, to_precision_bits
from olive.data.config import DataConfig
from olive.engine.config import RunPassConfig
from olive.hardware.accelerator import DEFAULT_CPU_ACCELERATOR, AcceleratorSpec
from olive.model import ModelConfig
from olive.package_config import OlivePackageConfig
from olive.passes.pass_config import BasePassConfig

logger = logging.getLogger(__name__)


class AutoOptimizerConfig(ConfigBase):
    precision: Union[Precision, PrecisionBits] = None
    accelerator: AcceleratorSpec = None

    # Should finetune?
    finetune: bool = True

    # data configs
    eval_data_config: Union[DataConfig, dict] = None
    train_data_config: Union[DataConfig, dict] = None
    calibration_data_config: Union[DataConfig, dict] = None

    # Convert to ONNX?
    convert_to_onnx: bool = True
    use_model_builder: bool = True
    use_dynamo_exporter: bool = False

    # Use quantization passes?
    quantize: bool = True
    quantization_algorithms: Optional[list[QuantAlgorithm]] = Field(default_factory=list)
    quantization_encodings: Optional[list[QuantEncoding]] = Field(default_factory=list)

    # Passes to exclude in generated config
    excluded_passes: Optional[list[str]] = Field(default_factory=list)

    # Write the config instead of running the passes.
    # Default config filename: auto_opt_config.json
    generate_config_only: bool = False


class AutoOptimizer:
    PASS_TYPES: ClassVar[dict[str, list[str]]] = {
        "pytorch_capture": ["CaptureSplitInfo"],
        "pytorch_finetune": ["DoRA", "LoftQ", "LoHa", "LoKr", "LoRA", "QLoRA"],
        "pytorch_quantize": ["QuaRot", "SpinQuant", "AutoAWQQuantizer", "GptqQuantizer"],
        "onnx_conversion": ["OnnxConversion", "ModelBuilder"],
        "onnx_peephole": ["OnnxPeepholeOptimizer"],
        "onnx_transformers": ["OrtTransformersOptimization"],
        "onnx_io_converter": ["OnnxIODataTypeConverter"],
        "onnx_prepare_qnn": [
            "DynamicToFixedShape",
            "QNNPreprocess",
            "MixedPrecisionOverrides",
        ],
        "onnx_surgeries": ["GraphSurgeries"],
        "onnx_quantize": [
            "OnnxBnb4Quantization",
            "OnnxMatMul4Quantizer",
            "OnnxDynamicQuantization",
            "OnnxStaticQuantization",
            "NVModelOptQuantization",
            "IncDynamicQuantization",
            "IncStaticQuantization",
        ],
        "onnx_finetune": ["OrtSessionParamsTuning"],
        "onnx_mnb_to_qdq": ["MatMulNBitsToQDQ"],
        "onnx_split_model": ["SplitModel"],
        "onnx_extract_adapters": ["ExtractAdapters"],
    }

    def __init__(
        self,
        model_config: ModelConfig,
        optimizer_config: Optional[AutoOptimizerConfig],
    ):
        self.model_config: ModelConfig = deepcopy(model_config)
        if self.model_config.type.lower() not in ["hfmodel", "onnxmodel", "pytorchmodel"]:
            raise ValueError(f"Unsupported input model type: {self.model_config.type}.")

        self.optimizer_config: AutoOptimizerConfig = deepcopy(optimizer_config) or AutoOptimizerConfig()
        if self.optimizer_config.finetune and not self.optimizer_config.train_data_config:
            raise ValueError("train_data_config is required for fine-tuning but none provided.")

        self.precision = optimizer_config.precision or Precision.FP32
        self.accelerator = deepcopy(optimizer_config.accelerator or DEFAULT_CPU_ACCELERATOR)
        self.excluded_passes = self.optimizer_config.excluded_passes or []
        self.excluded_passes = {_.lower() for _ in self.excluded_passes}
        self.quantization_algorithms = (
            set(deepcopy(self.optimizer_config.quantization_algorithms))
            if self.optimizer_config.quantization_algorithms
            else set(QuantAlgorithm)
        )
        self.quantization_encodings = (
            set(deepcopy(self.optimizer_config.quantization_encodings))
            if self.optimizer_config.quantization_encodings
            else set(QuantEncoding)
        )

    def generate_run_passes_configs(
        self, run_passes_configs: dict[str, Union[list[RunPassConfig], dict[str, Any]]] = None
    ) -> dict[str, list[type[BasePassConfig]]]:
        """Return a workflow config that can be used for search."""
        package_config = OlivePackageConfig.load_default_config()

        run_passes_configs = run_passes_configs or {}
        run_pass_config_by_type: dict[str, type[RunPassConfig]] = {}
        for value in run_passes_configs.values():
            pconfigs = value if isinstance(value, list) else [value]
            for pconfig in pconfigs:
                if isinstance(pconfig, RunPassConfig):
                    run_pass_config_by_type[pconfig.type.lower()] = pconfig.dict()
                else:
                    run_pass_config_by_type[pconfig["type"].lower()] = pconfig

        # Ideally, these exclusions should be part of the Pass.validate_config but current
        # logic doesn't allow for these checks to be at runtime. These checks are being enforced
        # as part of config validation i.e. pydantic validation.
        self._exclude_based_on_input_run_pass_configs(run_pass_config_by_type)

        substitutions = {
            "train_data_config": self.optimizer_config.train_data_config,
            "data_config": self.optimizer_config.calibration_data_config,
            "eval_data_config": self.optimizer_config.eval_data_config,
            "device": self.optimizer_config.accelerator.accelerator_type,
            "use_dynamo_exporter": self.optimizer_config.use_dynamo_exporter,
            "save_metadata_for_token_generation": self.optimizer_config.use_model_builder,
            "metadata_only": self.optimizer_config.use_model_builder,
            "precision": self.optimizer_config.precision,
            "bits": to_precision_bits(self.optimizer_config.precision),
        }

        resultant_run_pass_configs = {}
        for name, pass_types in self._included_pass_types.items():
            run_passes_configs = []
            for pass_type_name in pass_types:
                pass_type = pass_type_name.lower()

                if pass_type in self.excluded_passes:
                    logger.debug("Excluding pass %s because it's explicitly excluded.", pass_type)
                    continue

                if not self._filter(pass_type, package_config):
                    continue

                logger.debug("Generating config for pass_type=%s", pass_type)

                pass_module = package_config.import_pass_module(pass_type)
                default_pass_config = pass_module.default_config(self.accelerator)

                run_pass_config = run_pass_config_by_type.get(pass_type) or {}
                pass_config = deepcopy(run_pass_config.get("config")) or {}

                for key, value in substitutions.items():
                    if (key in default_pass_config) and (key not in pass_config):
                        pass_config[key] = value

                run_pass_config = RunPassConfig.parse_obj(
                    {
                        "type": pass_type,
                        "config": pass_module.generate_config(self.accelerator, pass_config, {}, False),
                        "host": run_pass_config.get("host"),
                        "evaluator": run_pass_config.get("evaluator"),
                    }
                )
                run_passes_configs.append(run_pass_config)

            if run_passes_configs:
                resultant_run_pass_configs[name] = run_passes_configs

        return resultant_run_pass_configs

    @property
    def _is_onnx_model(self) -> bool:
        return self.model_config.type == "onnxmodel"

    @property
    def _is_pytorch_model(self):
        return self.model_config.type in ["pytorchmodel", "hfmodel"]

    @property
    def _included_pass_types(self):
        pass_types = deepcopy(AutoOptimizer.PASS_TYPES)
        if self._is_pytorch_model:
            if not self.optimizer_config.finetune:
                pass_types.pop("pytorch_finetune", None)

            if not self.optimizer_config.quantize:
                pass_types.pop("pytorch_quantize", None)
        else:
            pass_types.pop("pytorch_capture", None)
            pass_types.pop("pytorch_finetune", None)
            pass_types.pop("pytorch_quantize", None)

        if self.optimizer_config.convert_to_onnx or self.optimizer_config.use_model_builder:
            pass_types["onnx_conversion"] = [
                "ModelBuilder" if self.optimizer_config.use_model_builder else "OnnxConversion"
            ]

        if self._is_onnx_model or self.optimizer_config.convert_to_onnx:
            if self.optimizer_config.use_model_builder:
                # Don't run optimizers when using model builder
                pass_types.pop("onnx_peephole", None)
                pass_types.pop("onnx_transformers", None)
                pass_types["onnx_quantize"].remove("OnnxMatMul4Quantizer")  # model already comes in int4

            if not self.optimizer_config.quantize:
                pass_types.pop("onnx_prepare_qnn", None)
                pass_types.pop("onnx_quantize", None)

            if not self.optimizer_config.finetune:
                pass_types.pop("onnx_finetune", None)
        else:
            if not self.optimizer_config.use_model_builder:
                pass_types.pop("onnx_conversion", None)

            pass_types.pop("onnx_peephole", None)
            pass_types.pop("onnx_transformers", None)
            pass_types.pop("onnx_io_converter", None)
            pass_types.pop("onnx_prepare_qnn", None)
            pass_types.pop("onnx_surgeries", None)
            pass_types.pop("onnx_quantize", None)
            pass_types.pop("onnx_finetune", None)
            pass_types.pop("onnx_mnb_to_qdq", None)
            pass_types.pop("onnx_split_model", None)
            pass_types.pop("onnx_extract_adapters", None)

        return pass_types

    def _exclude_based_on_input_run_pass_configs(self, run_pass_config_by_type: dict[str, dict[str, Any]]):
        run_pass_config = run_pass_config_by_type.get("dynamictofixedshape")
        if not (
            run_pass_config
            and ((run_pass_config.get("config") or {}).get("dim_param") is not None)
            and ((run_pass_config.get("config") or {}).get("dim_value") is not None)
        ):
            self.excluded_passes.add("dynamictofixedshape")
            logger.debug("Excluding pass DynamicToFixedShape because 'dim_param' & 'dim_value' aren't provided.")

        run_pass_config = run_pass_config_by_type.get("mixedprecisionoverrides")
        if not (run_pass_config and ((run_pass_config.get("config") or {}).get("overrides_config") is not None)):
            self.excluded_passes.add("mixedprecisionoverrides")
            logger.debug("Excluding pass MixedPrecisionOverrides because 'overrides_config' isn't provided.")

        run_pass_config = run_pass_config_by_type.get("graphsurgeries")
        if not (run_pass_config and ((run_pass_config.get("config") or {}).get("surgeries") is not None)):
            self.excluded_passes.add("graphsurgeries")
            logger.debug("Excluding pass GraphSurgeries because 'surgeries' isn't provided.")

    def _filter(self, pass_type: str, package_config: OlivePackageConfig):
        pass_module_config = package_config.get_pass_module_config(pass_type)

        # Exclude based on supported precision
        if self.precision not in pass_module_config.supported_precisions:
            logger.debug("Excluding pass %s because it doesn't support the selected precision.", pass_type)
            return False

        # Exclude based on supported execution provider
        if self.accelerator.execution_provider not in pass_module_config.supported_providers:
            logger.debug("Excluding pass %s because it doesn't support selected execution provider.", pass_type)
            return False

        # Exclude based on supported devices
        if self.accelerator.accelerator_type not in pass_module_config.supported_accelerators:
            logger.debug("Excluding pass %s because it doesn't support selected accelerator.", pass_type)
            return False

        # Exclude based on supported quantization algorithms
        if pass_module_config.supported_algorithms and not pass_module_config.supported_algorithms.intersection(
            self.quantization_algorithms
        ):
            logger.debug("Excluding pass %s because it doesn't support selected quantization algorithms.", pass_type)
            return False

        # Exclude based on supported quantization encoding
        if (
            pass_module_config.supported_quantization_encodings
            and not pass_module_config.supported_quantization_encodings.intersection(self.quantization_encodings)
        ):
            logger.debug("Excluding pass %s because it doesn't support selected quantization encodings.", pass_type)
            return False

        return True
