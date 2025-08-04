#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

import argparse
import copy
import os

from quark.shares.utils.log import ScreenLogger
from quark.torch.export import ExporterConfig, JsonExporterConfig, OnnxExporterConfig
from quark.torch.quantization import (
    AutoSmoothQuantConfig,
    AWQConfig,
    Config,
    FP8E4M3PerTensorSpec,
    Int4PerChannelSpec,
    ProgressiveSpec,
    QuantizationConfig,
    load_pre_optimization_config_from_file,
    load_quant_algo_config_from_file,
)

from olive.passes.quark_quantizer.torch.language_modeling.llm_ptq.customized_configuration import (
    DEPRECATED_QUANT_SCHEME,
    INT8_PER_TENSOR_DYNAMIC_SPEC,
    INT8_PER_TENSOR_SPEC,
    INT8_PER_TOKEN_DYNAMIC_SPEC,
    UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC,
    fp4_per_group_sym_spec,
    fp6_e2m3_per_group_sym_spec,
    fp6_e3m2_per_group_sym_spec,
    get_global_config,
    ocp_mxfp4_spec,
    ocp_mxfp6_e2m3_spec,
    ocp_mxfp6_e3m2_spec,
    ocp_mxfp8_e4m3_spec,
)
from olive.passes.quark_quantizer.torch.language_modeling.llm_utils.model_preparation import (
    MODEL_NAME_EXCLUDE_LAYERS_MAP,
    MODEL_NAME_KV_LAYERS_MAP,
    MODEL_NAME_Q_LAYERS_MAP,
    MOE_MODEL_NAME_EXPERTS_LAYERS_MAP,
)

EXAMPLES_DIR = os.path.abspath(os.path.dirname(__file__))
logger = ScreenLogger(__name__)

"""
Instructions for Setting Up Quark Quantization Configuration:
Step 1: Configure `QuantizationSpec` for torch.Tensors. Specify attributes such as dtype, observer_method, etc.
Step 2: Establish `QuantizationConfig` for nn.Module. Define the QuantizationSpec of input_tensors, output_tensors, weight, and bias.
Step 3: Set up the overall `Config` for the model. This includes:
        - global_quant_config (required)
        - layer_type_quant_config
        - layer_quant_config
        - kv_cache_quant_config
        - exclude
        - pre_quant_opt_config
        - algo_config
        - quant_mode
"""

# Step 1: Configure `DataTypeSpec` for torch.Tensors. Specify attributes such as dtype, observer_method, etc. More customer settings refer customized_configuration.py
FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()
FP8_PER_TENSOR_SPEC_DYNAMIC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True).to_quantization_spec()
INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()
MXFP8_PER_GROUP_SPEC = ocp_mxfp8_e4m3_spec(is_dynamic=True)


# Step 3: Set up the overall `Config` for the model.
def get_config(args: argparse.Namespace, model_type: str) -> Config:
    quant_scheme = args.quant_scheme
    group_size = args.group_size
    model_dir = args.model_dir
    kv_cache_dtype = args.kv_cache_dtype
    fp8_attention_quant = args.fp8_attention_quant
    moe_experts_second_step_config = args.moe_experts_second_step_config
    exclude_layers = args.exclude_layers
    pre_quantization_optimization = args.pre_quantization_optimization
    pre_optimization_config_file_path = args.pre_optimization_config_file_path
    quant_algo = args.quant_algo
    quant_algo_config_file_path = args.quant_algo_config_file_path
    group_size_per_layer = args.group_size_per_layer
    scale_format = args.scale_format
    scale_calculation_mode = args.scale_calculation_mode

    if quant_scheme in DEPRECATED_QUANT_SCHEME:
        logger.info(
            "[WARNING] The quantization scheme '%s' is deprecated and will be removed in the next AMD Quark release in favor of '%s'. "
            "Please use '--quant_scheme %s'.",
            quant_scheme,
            DEPRECATED_QUANT_SCHEME[quant_scheme],
            DEPRECATED_QUANT_SCHEME[quant_scheme],
        )

        quant_scheme = DEPRECATED_QUANT_SCHEME[quant_scheme]

    supported_kv_cache_type = {
        "fp8": FP8_PER_TENSOR_SPEC,
        "fp8_dynamic": FP8_PER_TENSOR_SPEC_DYNAMIC,
        "int8_per_tensor_static": INT8_PER_TENSOR_SPEC,
        "int8_per_tensor_dynamic": INT8_PER_TENSOR_DYNAMIC_SPEC,
        "int8_per_token": INT8_PER_TOKEN_DYNAMIC_SPEC,
        "uint4": UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC,
        "mxfp8": MXFP8_PER_GROUP_SPEC,
        "fp4_per_group": fp4_per_group_sym_spec(group_size, scale_format, None, True),
        "mxfp4": ocp_mxfp4_spec(scale_calculation_mode, True),
        "fp6e2m3_per_group": fp6_e2m3_per_group_sym_spec(group_size, scale_format, None, True),
        "mxfp6_e2m3": ocp_mxfp6_e2m3_spec(scale_calculation_mode, True),
        "fp6e3m2_per_group": fp6_e3m2_per_group_sym_spec(group_size, scale_format, None, True),
        "mxfp6_e3m2": ocp_mxfp6_e3m2_spec(scale_calculation_mode, True),
    }

    if kv_cache_dtype is not None and kv_cache_dtype not in supported_kv_cache_type:
        raise ValueError(f"The kv_cache_dtype='{kv_cache_dtype}' is not supported, only {supported_kv_cache_type} are.")

    global_quant_config = get_global_config(quant_scheme, group_size, scale_format, scale_calculation_mode)

    # Set up `layer_quant_config` and `kv_cache_quant_config`
    layer_quant_config = {}
    kv_cache_quant_config = {}
    if kv_cache_dtype is not None:
        kv_cache_spec = supported_kv_cache_type[kv_cache_dtype]

        if model_type not in MODEL_NAME_KV_LAYERS_MAP:
            raise ValueError(
                f"KV cache configuration of {model_type} could not be supported automaticly,"
                "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP"
            )

        kv_layers_name = MODEL_NAME_KV_LAYERS_MAP[model_type]
        for layer_name in kv_layers_name:
            kv_cache_quant_config[layer_name] = QuantizationConfig(
                input_tensors=global_quant_config.input_tensors,
                weight=global_quant_config.weight,
                output_tensors=kv_cache_spec,
            )
        layer_quant_config = kv_cache_quant_config.copy()

    group_size_per_layer = group_size_per_layer or []
    for layer, raw_group_size in group_size_per_layer:
        try:
            group_size = int(raw_group_size)
        except ValueError as err:
            raise ValueError(
                f"Invalid group size '{raw_group_size}' for layer '{layer}'. Group size must be an integer."
            ) from err
        layer_config = layer_quant_config.get(layer, copy.deepcopy(global_quant_config))
        layer_config.weight.group_size = group_size
        layer_quant_config[layer] = layer_config

    if fp8_attention_quant:
        attn_qspec = FP8_PER_TENSOR_SPEC

        if model_type not in MODEL_NAME_Q_LAYERS_MAP:
            raise ValueError(
                f"Q_proj configuration of {model_type} could not be supported automaticly,"
                "please add the q_proj layers in MODEL_NAME_Q_LAYERS_MAP"
            )

        q_layers_name = MODEL_NAME_Q_LAYERS_MAP[model_type]
        layer_quant_config[q_layers_name] = QuantizationConfig(
            input_tensors=global_quant_config.input_tensors,
            weight=global_quant_config.weight,
            output_tensors=attn_qspec,
        )
    else:
        attn_qspec = None

    #  Add MoE experts second step quantization configuration
    if moe_experts_second_step_config is not None:
        assert moe_experts_second_step_config in ["w_int4_per_channel_sym"], (
            "Currently, only w_int4_per_channel_sym is supported for MoE experts second step quantization, "
            "please add the MoE experts second step quantization configuration in quantize_quark.py"
        )

        assert model_type in MOE_MODEL_NAME_EXPERTS_LAYERS_MAP, (
            f"Currently, {model_type} is not supported for MoE experts second step quantization, "
            f"please add {model_type} model in MOE_MODEL_NAME_EXPERTS_LAYERS_MAP"
        )

        if moe_experts_second_step_config == "w_int4_per_channel_sym":
            weight_quant_spec = ProgressiveSpec(
                first_stage=global_quant_config.weight, second_stage=INT4_PER_CHANNEL_SPEC
            )
            experts_layers_name = MOE_MODEL_NAME_EXPERTS_LAYERS_MAP[model_type]

            for layer_name in experts_layers_name:
                layer_quant_config[layer_name] = QuantizationConfig(
                    input_tensors=global_quant_config.input_tensors,
                    weight=weight_quant_spec.to_quantization_spec(),
                    output_tensors=global_quant_config.output_tensors,
                )

    # Set up `exclude`
    if "c4ai-command-r-08-2024" in model_dir.lower():  # no quantization for particular layer
        MODEL_NAME_EXCLUDE_LAYERS_MAP["cohere"].append("*2.down_proj")

    if exclude_layers is None:
        if model_type in MODEL_NAME_EXCLUDE_LAYERS_MAP:
            exclude_layers_final = MODEL_NAME_EXCLUDE_LAYERS_MAP[model_type]
        else:
            exclude_layers_final = ["lm_head"]
            import warnings

            warnings.warn(
                f"Exclude layers configuration for {model_type} could not be supported automatically."
                'Using EXCLUDE_LAYERS = ["lm_head"]. Please customize the exclude layers in MODEL_NAME_EXCLUDE_LAYERS_MAP.',
                UserWarning,
            )
    else:
        exclude_layers_final = exclude_layers

    # Set up `pre_opt_config`
    pre_optimization_configs = []
    if "rotation" in pre_quantization_optimization:
        pre_optimization_config_file_path = (
            pre_optimization_config_file_path
            if pre_optimization_config_file_path
            else os.path.join(EXAMPLES_DIR, "models", model_type, "rotation_config.json")
        )
        pre_quant_opt_config = load_pre_optimization_config_from_file(pre_optimization_config_file_path)
        pre_optimization_configs.append(pre_quant_opt_config)
    if "quarot" in pre_quantization_optimization:
        pre_optimization_config_file_path = (
            pre_optimization_config_file_path
            if pre_optimization_config_file_path
            else os.path.join(EXAMPLES_DIR, "models", model_type, "quarot_config.json")
        )
        pre_quant_opt_config = load_pre_optimization_config_from_file(pre_optimization_config_file_path)
        pre_quant_opt_config.kv_cache_quant = kv_cache_dtype is not None
        pre_quant_opt_config.act_quant = global_quant_config.input_tensors is not None
        pre_optimization_configs.append(pre_quant_opt_config)
    if "smoothquant" in pre_quantization_optimization:
        pre_optimization_config_file_path = (
            pre_optimization_config_file_path
            if pre_optimization_config_file_path
            else os.path.join(EXAMPLES_DIR, "models", model_type, "smooth_config.json")
        )
        pre_opt_config = load_pre_optimization_config_from_file(pre_optimization_config_file_path)
        pre_optimization_configs.append(pre_opt_config)

        smoothquant_alpha = pre_opt_config.alpha
        if global_quant_config.input_tensors is None and smoothquant_alpha > 0:
            logger.info(
                "[WARNING] Weight-only quantization is used, but SmoothQuant alpha=%s is larger than 0.0. In this case, using alpha = 0.0 is recommended to shift all the quantization difficulty from the weights into the activations.",
                smoothquant_alpha,
            )

        if global_quant_config.weight is None and smoothquant_alpha < 1:
            logger.info(
                "[WARNING] Activation-only quantization is used, but SmoothQuant alpha=%s is smaller than 1.0. In this case, using alpha = 1.0 is recommended to shift all the quantization difficulty from the activations into the weights.",
                smoothquant_alpha,
            )

        if (
            global_quant_config.weight is not None
            and global_quant_config.input_tensors is not None
            and smoothquant_alpha in [0.0, 1.0]
        ):
            logger.info(
                "[WARNING] Both weights and activations are quantized, but SmoothQuant alpha=%s is used. alpha = 0.0 shifts all the quantization difficulty to activations, while alpha = 1.0 shifts all the quantization difficulty to the weights. If this is the desired behavior, this warning can be ignored.",
                smoothquant_alpha,
            )

    # Set up `algo_config`
    algo_config = (
        load_algo_config(quant_algo, quant_scheme, quant_algo_config_file_path, model_type) if quant_algo else None
    )

    return Config(
        global_quant_config=global_quant_config,
        layer_quant_config=layer_quant_config,
        kv_cache_quant_config=kv_cache_quant_config,
        softmax_quant_spec=attn_qspec,
        exclude=exclude_layers_final,
        pre_quant_opt_config=pre_optimization_configs,
        algo_config=algo_config,
    )


def load_algo_config(quant_algo, quant_scheme, quant_algo_config_file_path, model_type):
    default_algo_config_file = None
    if quant_algo == "awq":
        default_algo_config_file = os.path.join(EXAMPLES_DIR, "models", model_type, "awq_config.json")
    elif quant_algo == "autosmoothquant":
        default_algo_config_file = os.path.join(EXAMPLES_DIR, "models", model_type, "autosmoothquant_config.json")
    elif quant_algo == "gptq":
        if quant_scheme not in ["w_uint4_per_group_asym", "w_uint4_per_channel_asym", "w_mxfp4_a_mxfp4"]:
            logger.warning(
                "GPTQ is only tested with uint4_per_group, w_uint4_per_channel_asym, and w_mxfp4_a_mxfp4 quantization in Quark, be careful to apply GPTQ on %s.",
                quant_scheme,
            )
        default_algo_config_file = os.path.join(EXAMPLES_DIR, "models", model_type, "gptq_config.json")
    quant_algo_config_file_path = (
        quant_algo_config_file_path if quant_algo_config_file_path else default_algo_config_file
    )
    if os.path.exists(quant_algo_config_file_path):
        algo_config = load_quant_algo_config_from_file(quant_algo_config_file_path)
    else:
        if quant_algo == "awq":
            algo_config = AWQConfig()
        elif quant_algo == "autosmoothquant":
            algo_config = AutoSmoothQuantConfig()
        else:
            raise ValueError("Missing quantization algorithm configuration")
    return algo_config


MERGE_REALQ_CONFIG = JsonExporterConfig(
    weight_merge_groups=[["*up_proj", "*gate_proj"], ["*q_proj", "*k_proj", "*v_proj"]],
    weight_format="real_quantized",
    pack_method="reorder",
)

NO_MERGE_REALQ_CONFIG = JsonExporterConfig(weight_format="real_quantized", pack_method="reorder")


def get_export_config(args: argparse.Namespace, model_type: str) -> ExporterConfig:
    export_config = None
    if args.weight_matrix_merge is True:
        export_config = ExporterConfig(json_export_config=MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig())
    else:
        export_config = ExporterConfig(
            json_export_config=NO_MERGE_REALQ_CONFIG, onnx_export_config=OnnxExporterConfig()
        )

    if args.kv_cache_dtype is not None:
        if model_type not in MODEL_NAME_KV_LAYERS_MAP:
            raise ValueError(
                f"KV cache configuration of {model_type} could not be supported automaticly,"
                "please add the KV layers in MODEL_NAME_KV_LAYERS_MAP"
            )
        export_config.json_export_config.kv_cache_group = MODEL_NAME_KV_LAYERS_MAP[model_type]

    if args.pack_method == "order":
        export_config.json_export_config.pack_method = "order"

    export_config.json_export_config.min_kv_scale = args.min_kv_scale

    return export_config
