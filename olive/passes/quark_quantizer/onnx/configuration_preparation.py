#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import logging
from typing import Any

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
    BFloat16Spec,
    BFP16Spec,
    CalibMethod,
    Int8Spec,
    Int16Spec,
    Int32Spec,
    QLayerConfig,
    QuantGranularity,
    ScaleType,
    UInt8Spec,
    UInt16Spec,
    UInt32Spec,
    XInt8Spec,
)

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


data_type_mapping = {
    "Int8": Int8Spec,
    "UInt8": UInt8Spec,
    "XInt8": XInt8Spec,
    "Int16": Int16Spec,
    "UInt16": UInt16Spec,
    "Int32": Int32Spec,
    "UInt32": UInt32Spec,
    "BFloat16": BFloat16Spec,
    "BFP16": BFP16Spec,
}


def get_global_config(global_config_dict: dict[str, Any]) -> QLayerConfig:
    activation_spec = UInt8Spec()
    if isinstance(global_config_dict, dict) and "activation" in global_config_dict:
        if "symmetric" in global_config_dict["activation"]:
            activation_spec.set_symmetric(global_config_dict["activation"]["symmetric"])
        if "scale_type" in global_config_dict["activation"]:
            activation_spec.set_scale_type(scale_type_mapping[global_config_dict["activation"]["scale_type"]])
        if "calibration_method" in global_config_dict["activation"]:
            activation_spec.set_calibration_method(
                calibration_method_mapping[global_config_dict["activation"]["calibration_method"]]
            )
        if "quant_granularity" in global_config_dict["activation"]:
            activation_spec.set_quant_granularity(
                quant_granularity_mapping[global_config_dict["activation"]["quant_granularity"]]
            )
        if "data_type" in global_config_dict["activation"]:
            activation_spec.set_data_type(data_type_mapping[global_config_dict["activation"]["data_type"]]().data_type)

    weight_spec = Int8Spec()
    if isinstance(global_config_dict, dict) and "weight" in global_config_dict:
        if "symmetric" in global_config_dict["weight"]:
            weight_spec.set_symmetric(global_config_dict["weight"]["symmetric"])
        if "scale_type" in global_config_dict["weight"]:
            weight_spec.set_scale_type(scale_type_mapping[global_config_dict["weight"]["scale_type"]])
        if "calibration_method" in global_config_dict["weight"]:
            weight_spec.set_calibration_method(
                calibration_method_mapping[global_config_dict["weight"]["calibration_method"]]
            )
        if "quant_granularity" in global_config_dict["weight"]:
            weight_spec.set_quant_granularity(
                quant_granularity_mapping[global_config_dict["weight"]["quant_granularity"]]
            )
        if "data_type" in global_config_dict["weight"]:
            weight_spec.set_data_type(data_type_mapping[global_config_dict["weight"]["data_type"]]().data_type)

    return QLayerConfig(activation=activation_spec, weight=weight_spec)


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
