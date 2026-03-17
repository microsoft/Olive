#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import copy
import logging
from typing import Any

from quark.onnx import ExtendedQuantType, QuantType
from quark.onnx.quantization.config.algorithm import (
    AdaQuantConfig,
    AdaRoundConfig,
    AlgoConfig,
    AutoMixprecisionConfig,
    BiasCorrectionConfig,
    CLEConfig,
    GPTQConfig,
    QuarotConfig,
    SmoothQuantConfig,
)
from quark.onnx.quantization.config.spec import (
    CalibMethod,
    QLayerConfig,
    QuantGranularity,
    ScaleType,
)

try:
    from quark.onnx.quantization.config.data_type import parse_data_type
except ImportError:
    parse_data_type = None

logger = logging.getLogger(__name__)


scale_type_mapping = {
    "Float32": ScaleType.Float32,
    "PowerOf2": ScaleType.PowerOf2,
    "Int16": ScaleType.Int16,
}


calibration_method_mapping = {
    "MinMax": CalibMethod.MinMax,
    "Entropy": CalibMethod.Entropy,
    "Percentile": CalibMethod.Percentile,
    "Distribution": CalibMethod.Distribution,
    "MinMSE": CalibMethod.MinMSE,
    "LayerwisePercentile": CalibMethod.LayerwisePercentile,
}


quant_granularity_mapping = {
    "Tensor": QuantGranularity.Tensor,
    "Channel": QuantGranularity.Channel,
    "Group": QuantGranularity.Group,
}


def convert_tensor_config(tensor_config_dict: dict[str, Any]) -> None:
    """Convert the old format string to the enum string."""
    if "symmetric" not in tensor_config_dict:
        tensor_config_dict["symmetric"] = True

    if "scale_type" not in tensor_config_dict:
        obj = ScaleType.Float32
    elif tensor_config_dict["scale_type"] not in scale_type_mapping:
        logger.warning("The scale type %s is not supported.", tensor_config_dict["scale_type"])
        obj = ScaleType.Float32
    else:
        obj = scale_type_mapping[tensor_config_dict["scale_type"]]
    tensor_config_dict["scale_type"] = f"{obj.__class__.__name__}.{obj.name}"

    if "calibration_method" not in tensor_config_dict:
        obj = CalibMethod.Percentile
    elif tensor_config_dict["calibration_method"] not in calibration_method_mapping:
        logger.warning("The calibration method %s is not supported.", tensor_config_dict["calibration_method"])
        obj = CalibMethod.Percentile
    else:
        obj = calibration_method_mapping[tensor_config_dict["calibration_method"]]
    tensor_config_dict["calibration_method"] = f"{obj.__class__.__name__}.{obj.name}"

    if "quant_granularity" not in tensor_config_dict:
        obj = QuantGranularity.Tensor
    elif tensor_config_dict["quant_granularity"] not in quant_granularity_mapping:
        logger.warning("The quant granularity %s is not supported.", tensor_config_dict["quant_granularity"])
        obj = QuantGranularity.Tensor
    else:
        obj = quant_granularity_mapping[tensor_config_dict["quant_granularity"]]
    tensor_config_dict["quant_granularity"] = f"{obj.__class__.__name__}.{obj.name}"

    if "data_type" not in tensor_config_dict:
        tensor_config_dict["data_type"] = "Int8"
    elif parse_data_type and parse_data_type(tensor_config_dict["data_type"]) is None:
        logger.warning("The data type %s is not supported.", tensor_config_dict["data_type"])
        tensor_config_dict["data_type"] = "Int8"


def convert_layer_config(layer_config_dict: dict[str, Any]) -> None:
    """Convert the old format string to the enum string in each item of the config."""
    if "activation" in layer_config_dict:
        convert_tensor_config(layer_config_dict["activation"])

    if "weight" in layer_config_dict:
        convert_tensor_config(layer_config_dict["weight"])

    if "bias" in layer_config_dict:
        convert_tensor_config(layer_config_dict["bias"])

    if "input_tensors" in layer_config_dict:
        convert_tensor_config(layer_config_dict["input_tensors"])

    if "output_tensors" in layer_config_dict:
        convert_tensor_config(layer_config_dict["output_tensors"])


def get_global_config(global_config_dict: dict[str, Any]) -> None:
    """Get the global configuration for the given global configuration dictionary."""
    if global_config_dict is None:
        return None
    elif not isinstance(global_config_dict, dict):
        raise ValueError("The global configuration should be a dictionary.")

    copied_layer_config = copy.deepcopy(global_config_dict)
    convert_layer_config(copied_layer_config)

    # This is for the backward compatibility
    if "activation" in copied_layer_config and "input_tensors" not in copied_layer_config:
        copied_layer_config["input_tensors"] = copied_layer_config["activation"]

    return QLayerConfig.from_dict(copied_layer_config)


def convert_layer_config_list_to_dict(
    layer_config_list: list[list[dict[str, Any], list[str]]],
) -> dict[QLayerConfig, list[str]]:
    """Convert the layer configuration list to a dictionary."""
    layer_config_dict: dict[QLayerConfig, list[str]] = {}

    for layer_config_item in layer_config_list:
        if not isinstance(layer_config_item, list) or len(layer_config_item) != 2:
            logger.warning("The element of layer configuration should be a list of two elements.")
            continue

        copied_layer_config = copy.deepcopy(layer_config_item[0])
        convert_layer_config(copied_layer_config)

        layer_config = QLayerConfig.from_dict(copied_layer_config)
        layer_config_dict[layer_config] = layer_config_item[1]

    return layer_config_dict


def get_layer_type_config(
    layer_type_config_list: list[list[str, dict[str, Any]]] | None,
) -> dict[QLayerConfig, list[str]] | None:
    """Get the layer type configuration for the given layer type configuration list."""
    if layer_type_config_list is None:
        return None
    elif not isinstance(layer_type_config_list, list):
        raise ValueError("The layer type configuration should be a list.")

    return convert_layer_config_list_to_dict(layer_type_config_list)


def get_specific_layer_config(
    specific_layer_config_list: list[list[str, dict[str, Any]]] | None,
) -> dict[QLayerConfig, list[str]] | None:
    """Get the specific layer configuration for the given specific layer configuration list."""
    if specific_layer_config_list is None:
        return None
    elif not isinstance(specific_layer_config_list, list):
        raise ValueError("The specific layer configuration should be a list.")

    return convert_layer_config_list_to_dict(specific_layer_config_list)


algorithm_mapping = {
    "smooth_quant": SmoothQuantConfig,
    "cle": CLEConfig,
    "bias_correction": BiasCorrectionConfig,
    "gptq": GPTQConfig,
    "auto_mixprecision": AutoMixprecisionConfig,
    "adaround": AdaRoundConfig,
    "adaquant": AdaQuantConfig,
    "quarot": QuarotConfig,
}


def update_algo_config(algo_config: AlgoConfig, config_dict: dict[str, Any]) -> None:
    """Update the algorithm configuration for the given algorithm configuration dictionary."""
    if isinstance(algo_config, (AdaRoundConfig, AdaQuantConfig)):
        if "optim_device" in config_dict:
            algo_config.optim_device = config_dict["optim_device"]
        if "infer_device" in config_dict:
            algo_config.infer_device = config_dict["infer_device"]
        if "fixed_seed" in config_dict:
            algo_config.fixed_seed = config_dict["fixed_seed"]
        if "data_size" in config_dict:
            algo_config.data_size = config_dict["data_size"]
        if "batch_size" in config_dict:
            algo_config.batch_size = config_dict["batch_size"]
        if "num_batches" in config_dict:
            algo_config.num_batches = config_dict["num_batches"]
        if "num_iterations" in config_dict:
            algo_config.num_iterations = config_dict["num_iterations"]
        if "learning_rate" in config_dict:
            algo_config.learning_rate = config_dict["learning_rate"]
        if "early_stop" in config_dict:
            algo_config.early_stop = config_dict["early_stop"]
        if "output_index" in config_dict:
            algo_config.output_index = config_dict["output_index"]
        if "lr_adjust" in config_dict:
            algo_config.lr_adjust = tuple(config_dict["lr_adjust"])
        if "target_op_type" in config_dict:
            algo_config.target_op_type = config_dict["target_op_type"]
        if "selective_update" in config_dict:
            algo_config.selective_update = config_dict["selective_update"]
        if "update_bias" in config_dict:
            algo_config.update_bias = config_dict["update_bias"]
        if "output_qdq" in config_dict:
            algo_config.output_qdq = config_dict["output_qdq"]
        if "drop_ratio" in config_dict:
            algo_config.drop_ratio = config_dict["drop_ratio"]
        if "mem_opt_level" in config_dict:
            algo_config.mem_opt_level = config_dict["mem_opt_level"]
        if "cache_dir" in config_dict:
            algo_config.cache_dir = config_dict["cache_dir"]
        if "log_period" in config_dict:
            algo_config.log_period = config_dict["log_period"]
        if "ref_model_path" in config_dict:
            algo_config.ref_model_path = config_dict["ref_model_path"]
        if "dynamic_batch" in config_dict:
            algo_config.dynamic_batch = config_dict["dynamic_batch"]
        if "parallel" in config_dict:
            algo_config.parallel = config_dict["parallel"]
        if "reg_param" in config_dict:
            algo_config.reg_param = config_dict["reg_param"]
        if "beta_range" in config_dict:
            algo_config.beta_range = tuple(config_dict["beta_range"])
        if "warm_start" in config_dict:
            algo_config.warm_start = config_dict["warm_start"]

    elif isinstance(algo_config, CLEConfig):
        if "cle_balance_method" in config_dict:
            algo_config.cle_balance_method = config_dict["cle_balance_method"]
        if "cle_steps" in config_dict:
            algo_config.cle_steps = config_dict["cle_steps"]
        if "cle_weight_threshold" in config_dict:
            algo_config.cle_weight_threshold = config_dict["cle_weight_threshold"]
        if "cle_scale_append_bias" in config_dict:
            algo_config.cle_scale_append_bias = config_dict["cle_scale_append_bias"]
        if "cle_scale_use_threshold" in config_dict:
            algo_config.cle_scale_use_threshold = config_dict["cle_scale_use_threshold"]
        if "cle_total_layer_diff_threshold" in config_dict:
            algo_config.cle_total_layer_diff_threshold = config_dict["cle_total_layer_diff_threshold"]

    elif isinstance(algo_config, SmoothQuantConfig):
        if "alpha" in config_dict:
            algo_config.alpha = config_dict["alpha"]

    else:
        # TODO(Gengxin): Configure the rest algorithms
        pass


def get_algo_config(algo_config_list: list[dict[str, Any]] | None) -> list[AlgoConfig]:
    """Get the algorithm configuration for the given algorithm configuration list."""
    algo_configs: list[AlgoConfig] = []

    if algo_config_list is None:
        return algo_configs

    for config_dict in algo_config_list:
        if "name" not in config_dict:
            logger.warning("Unknown algorithm configuration. Ignoring.")
            continue

        if config_dict["name"] not in algorithm_mapping:
            logger.warning("Unsupported algorithm %s. Ignoring.", config_dict["name"])
            continue

        algo_config = algorithm_mapping[config_dict["name"]]()
        update_algo_config(algo_config, config_dict)
        algo_configs.append(algo_config)

    return algo_configs


quant_type_mapping = {
    "Int8": QuantType.QInt8,
    "UInt8": QuantType.QUInt8,
    "Int16": QuantType.QInt16,
    "UInt16": QuantType.QUInt16,
    "Int32": ExtendedQuantType.QInt32,
    "UInt32": ExtendedQuantType.QUInt32,
    "Float16": ExtendedQuantType.QFloat16,
    "BFloat16": ExtendedQuantType.QBFloat16,
    "BFP16": ExtendedQuantType.QBFP,
    "BFP": ExtendedQuantType.QBFP,
    "MX": ExtendedQuantType.QMX,
}


def get_extra_options(extra_options_dict: dict[str, Any] | None) -> dict[str, Any]:
    """Get the extra options for the given extra options dictionary."""
    extra_options: dict[str, Any] = {}

    if extra_options_dict is None:
        return extra_options

    for key, value in extra_options_dict.items():
        # This option is deprecated and will be removed in the next AMD Quark release
        if key == "MixedPrecisionTensor":
            if not isinstance(value, dict):
                logger.warning("The value for extra option 'MixedPrecisionTensor' should be a dict.")
                continue

            mixed_precision_tensor = {}
            for k, v in value.items():
                quant_type = quant_type_mapping[k]
                mixed_precision_tensor[quant_type] = v
            extra_options[key] = mixed_precision_tensor

        # This option is for standard tensor-wise mixed precision quantization
        elif key == "TensorQuantOverrides":
            if not isinstance(value, dict):
                logger.warning("The value for extra option 'TensorQuantOverrides' should be a dict.")
                continue

            extra_options[key] = copy.deepcopy(value)
            for tensor_name, quant_overrides in extra_options[key].items():
                if not isinstance(quant_overrides, list):
                    logger.warning("The quant overrides for %s should be a list.", tensor_name)
                    continue

                for quant_override in quant_overrides:
                    if not isinstance(quant_override, dict):
                        logger.warning("The quant override %s should be a dict.", quant_override)
                        continue

                    if quant_override["quant_type"] not in quant_type_mapping:
                        logger.warning("The quant type %s is not supported.", quant_override["quant_type"])
                        continue

                    quant_override["quant_type"] = quant_type_mapping[quant_override["quant_type"]]

        # Other options are kept as is
        else:
            extra_options[key] = value

    return extra_options
