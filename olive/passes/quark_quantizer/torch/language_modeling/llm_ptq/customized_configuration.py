#
# Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from quark.torch.quantization import (
    Bfloat16Spec,
    BFP16Spec,
    Float16Spec,
    FP4PerGroupSpec,
    FP6E2M3PerGroupSpec,
    FP6E3M2PerGroupSpec,
    FP8E4M3PerChannelSpec,
    FP8E4M3PerGroupSpec,
    FP8E4M3PerTensorSpec,
    FP8E5M2PerGroupSpec,
    Int2PerGroupSpec,
    Int4PerChannelSpec,
    Int4PerGroupSpec,
    Int4PerTensorSpec,
    Int8PerChannelSpec,
    Int8PerGroupSpec,
    Int8PerTensorSpec,
    MX6Spec,
    OCP_MXFP4DiffsSpec,
    OCP_MXFP4Spec,
    OCP_MXFP6E2M3Spec,
    OCP_MXFP6E3M2Spec,
    OCP_MXFP8E4M3Spec,
    OCP_MXFP8E5M2Spec,
    OCP_MXINT8Spec,
    QuantizationConfig,
    ScaleQuantSpec,
    Uint4PerChannelSpec,
    Uint4PerGroupSpec,
    Uint8PerGroupSpec,
    Uint8PerTensorSpec,
)

DEPRECATED_QUANT_SCHEME = {
    "w_mx_fp4_a_mx_fp4_sym": "w_mxfp4_a_mxfp4",
    "w_mx_fp6_e3m2_sym": "w_mxfp6_e3m2",
    "w_mx_fp6_e2m3_sym": "w_mxfp6_e2m3",
    "w_mx_int8_per_group_sym": "w_mxint8",
    "w_mxfp4_a_mxfp4_sym": "w_mxfp4_a_mxfp4",
    "w_mx_fp6_e2m3_a_mx_fp6_e2m3": "w_mxfp6_e2m3_a_mxfp6_e2m3",
    "w_mx_fp6_e3m2_a_mx_fp6_e3m2": "w_mxfp6_e3m2_a_mxfp6_e3m2",
    "w_mx_fp4_a_mx_fp6_sym": "w_mxfp4_a_mxfp6",
    "w_mx_fp8_a_mx_fp8": "w_mxfp8_a_mxfp8",
}

OCP_MX_QUANT_SCHEME = [
    "w_mx6",
    "w_mx6_a_mx6",
    "w_mxfp4",
    "w_mxfp4_a_mxfp4",
    "w_mxfp4_a_mxfp6",
    "w_mxfp4_a_mxfp8",
    "w_mxfp4_diffs",
    "w_mxfp6_e2m3_a_mxfp6_e2m3",
    "w_mxfp6_e2m3",
    "w_mxfp6_e3m2_a_mxfp6_e3m2",
    "w_mxfp6_e3m2",
    "w_mxfp8",
    "w_mxfp8_a_mxfp8",
    "w_mxint8",
]


SUPPORTED_QUANT_SCHEME = [
    *[
        "w_bfp16",
        "w_bfp16_a_bfp16",
        "w_bfp16_per_group_sym",
        "w_fp4_a_fp4_scale_fp8",
        "w_fp4_per_group",
        "w_fp4_per_group_a_fp4_per_group",
        "w_fp4_per_group_a_fp8_e4m3_per_group",
        "w_fp4_scale_fp8",
        "w_fp6_e2m3_per_group_a_fp6_e2m3_per_group",
        "w_fp6_e3m2_per_group_a_fp6_e3m2_per_group",
        "w_fp8_a_fp8",
        "w_fp8_a_fp8_o_fp8",
        "w_fp8_per_channel_sym",
        "w_int2_per_group_asym",
        "w_int4_per_channel_asym",
        "w_int4_per_channel_sym",
        "w_int4_per_group_asym",
        "w_int4_per_group_sym",
        "w_int8_a_int8_per_tensor_sym",
        "w_int8_a_int8_per_tensor_sym_dynamic",
        "w_int8_a_int8_per_token_dynamic",
        "w_int8_per_channel_a_int8_per_tensor_sym",
        "w_int8_per_channel_a_int8_per_tensor_sym_dynamic",
        "w_int8_per_group_sym",
        "w_int8_per_tensor_mse",
        "w_int8_per_tensor_percentile",
        "w_int8_per_tensor_sym",
        "w_uint4_a_bfloat16_per_group_asym",
        "w_uint4_a_uint4_per_channel",
        "w_uint4_per_channel_a_int8_per_tensor",
        "w_uint4_per_channel_asym",
        "w_uint4_per_channel_sym",
        "w_uint4_per_group_a_int8_per_tensor",
        "w_uint4_per_group_sym",
        "w_uint4_per_group_asym",
        "w_uint4_per_token_a_int8_per_channel",
        "w_uint8_a_uint8_per_tensor_asym",
        "w_uint8_per_group_asym",
        "w_uint8_per_tensor_mse",
        "w_uint8_per_tensor_percentile",
    ],
    *OCP_MX_QUANT_SCHEME,
    *list(DEPRECATED_QUANT_SCHEME),
]

FLOAT16_SPEC = Float16Spec().to_quantization_spec()

BFLOAT16_SPEC = Bfloat16Spec().to_quantization_spec()

FP8_PER_TENSOR_SPEC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False).to_quantization_spec()

FP8_PER_TENSOR_SPEC_DYNAMIC = FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True).to_quantization_spec()

INT4_PER_TENSOR_SPEC = Int4PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()


INT8_PER_CHANNEL_SPEC = Int8PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()


UINT4_PER_GROUP_ASYM_SPEC = Uint4PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

INT8_PER_TENSOR_SPEC = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

UINT8_PER_TENSOR_ASPEC = Uint8PerTensorSpec(
    observer_method="min_max", symmetric=False, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

INT8_PER_TOKEN_DYNAMIC_SPEC = Int8PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=True
).to_quantization_spec()

INT8_PER_TENSOR_DYNAMIC_SPEC = Int8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=True
).to_quantization_spec()

INT8_PER_GROUP_SYM_SPEC = Int8PerGroupSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

UINT8_PER_GROUP_ASYM_SPEC = Uint8PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

OCP_MXFP8_STATIC_SPEC = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=False).to_quantization_spec()

OCP_MXFP8_DYNAMIC_SPEC = OCP_MXFP8E4M3Spec(ch_axis=-1, is_dynamic=True).to_quantization_spec()

W_BFP16_SPEC = BFP16Spec(is_dynamic=False, ch_axis=-1).to_quantization_spec()

A_BFP16_SPEC = BFP16Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()

W_MX6_SPEC = MX6Spec(ch_axis=-1, block_size=32, is_dynamic=False).to_quantization_spec()

A_MX6_SPEC = MX6Spec(ch_axis=-1, block_size=32, is_dynamic=True).to_quantization_spec()

# Data type spec for testing, not applied on the specific backend
UINT8_PER_TENSOR_SPEC = Uint8PerTensorSpec(
    observer_method="min_max", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

FP8_PER_CHANNEL_SPEC = FP8E4M3PerChannelSpec(is_dynamic=False, ch_axis=0).to_quantization_spec()

INT4_PER_CHANNEL_ASYM_SPEC = Int4PerChannelSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()

INT4_PER_GROUP_ASYM_SPEC = Int4PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

UINT4_PER_GROUP_SYM_SPEC = Uint4PerGroupSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

UINT4_PER_CHANNEL_SYM_SPEC = Uint4PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()

UINT4_PER_CHANNEL_ASYM_SPEC = Uint4PerChannelSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()

UINT4_PER_CHANNEL_ASYM_DYNAMIC_SPEC = Uint4PerChannelSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=True
).to_quantization_spec()

INT8_PER_TENSOR_PERCENTILE_SPEC = Int8PerTensorSpec(
    observer_method="percentile", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

INT8_PER_TENSOR_MSE_SPEC = Int8PerTensorSpec(
    observer_method="MSE", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

UINT8_PER_TENSOR_PERCENTILE_SPEC = Uint8PerTensorSpec(
    observer_method="percentile", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

UINT8_PER_TENSOR_MSE_SPEC = Uint8PerTensorSpec(
    observer_method="MSE", symmetric=True, scale_type="float", round_method="half_even", is_dynamic=False
).to_quantization_spec()

FP4_PER_GROUP_SYM_STATIC_GS16_SPEC = FP4PerGroupSpec(
    ch_axis=-1, group_size=16, is_dynamic=False, scale_format="e8m0", scale_calculation_mode="even"
).to_quantization_spec()

FP4_PER_GROUP_SYM_DYNAMIC_GS16_SPEC = FP4PerGroupSpec(
    ch_axis=-1, group_size=16, is_dynamic=True, scale_format="e8m0", scale_calculation_mode="even"
).to_quantization_spec()

OCP_MX_SEPERATED_FP4_PER_GROUP_DIFFS_SPEC = OCP_MXFP4DiffsSpec(ch_axis=-1, is_dynamic=False).to_quantization_spec()

# Two stage configurations
# This configuration specifies that the first stage is a per-group FP4 quantization, and the second stage is a per-tensor FP8 quantization.
# The first stage is used to quantize the input tensor, and the second stage is used to quantize the scale of the first stage.
FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC = ScaleQuantSpec(
    first_stage=FP4PerGroupSpec(group_size=16, is_dynamic=False),
    second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=False),
).to_quantization_spec()

FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_DYNAMIC = ScaleQuantSpec(
    first_stage=FP4PerGroupSpec(group_size=16, is_dynamic=True),
    second_stage=FP8E4M3PerTensorSpec(observer_method="min_max", is_dynamic=True),
).to_quantization_spec()


def ocp_mxfp6_e2m3_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXFP6E2M3Spec(
        ch_axis=-1, scale_calculation_mode=scale_calculation_mode, is_dynamic=is_dynamic
    ).to_quantization_spec()


def ocp_mxfp6_e3m2_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXFP6E3M2Spec(
        ch_axis=-1, scale_calculation_mode=scale_calculation_mode, is_dynamic=is_dynamic
    ).to_quantization_spec()


def ocp_mxfp4_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXFP4Spec(
        ch_axis=-1, is_dynamic=is_dynamic, scale_calculation_mode=scale_calculation_mode
    ).to_quantization_spec()


def ocp_mxfp8_e4m3_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXFP8E4M3Spec(
        scale_calculation_mode=scale_calculation_mode, ch_axis=-1, is_dynamic=is_dynamic
    ).to_quantization_spec()


def ocp_mxfp8_e5m2_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXFP8E5M2Spec(
        scale_calculation_mode=scale_calculation_mode, ch_axis=-1, is_dynamic=is_dynamic
    ).to_quantization_spec()


def ocp_mxint8_spec(scale_calculation_mode="even", is_dynamic=True):
    return OCP_MXINT8Spec(
        scale_calculation_mode=scale_calculation_mode, ch_axis=-1, is_dynamic=is_dynamic
    ).to_quantization_spec()


# FP per-group config
def fp8_e4m3_per_group_sym_spec(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP8E4M3PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic,
    ).to_quantization_spec()


def fp8_e5m2_per_group_sym_spec(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP8E5M2PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic,
    ).to_quantization_spec()


def fp4_per_group_sym_spec(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP4PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic,
    ).to_quantization_spec()


def fp6_e2m3_per_group_sym_spec(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP6E2M3PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic,
    ).to_quantization_spec()


def fp6_e3m2_per_group_sym_spec(group_size, scale_format="e8m0", scale_calculation_mode="even", is_dynamic=True):
    return FP6E3M2PerGroupSpec(
        ch_axis=-1,
        group_size=group_size,
        scale_format=scale_format,
        scale_calculation_mode=scale_calculation_mode,
        is_dynamic=is_dynamic,
    ).to_quantization_spec()


# BFloat16 config
BFP16_PER_GROUP_SYM_SPEC = BFP16Spec(is_dynamic=True, ch_axis=-1).to_quantization_spec()


# Float16 config
FLOAT16_CONFIG = QuantizationConfig(input_tensors=FLOAT16_SPEC, weight=FLOAT16_SPEC)

W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC
)

# Int per tensor config
W_INT4_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT4_PER_TENSOR_SPEC)

W_INT8_PER_TENSOR_CONFIG = QuantizationConfig(weight=INT8_PER_TENSOR_SPEC)

W_INT8_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_SPEC)

W_UINT8_A_UINT8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=UINT8_PER_TENSOR_ASPEC, weight=UINT8_PER_TENSOR_ASPEC
)

W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_DYNAMIC_SPEC, weight=INT8_PER_TENSOR_DYNAMIC_SPEC
)

# Int per Channel Config
W_INT8_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT8_PER_CHANNEL_SPEC)

W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_CHANNEL_SPEC
)

W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_DYNAMIC_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_DYNAMIC_SPEC, weight=INT8_PER_CHANNEL_SPEC
)

# Int per Group Config
W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG = QuantizationConfig(input_tensors=BFLOAT16_SPEC, weight=UINT4_PER_GROUP_ASYM_SPEC)

W_INT8_PER_GROUP_CONFIG = QuantizationConfig(weight=INT8_PER_GROUP_SYM_SPEC)

W_UINT8_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT8_PER_GROUP_ASYM_SPEC)

W_MXFP8_CONFIG = QuantizationConfig(weight=OCP_MXFP8_STATIC_SPEC)
W_MXFP8_A_MXFP8_CONFIG = QuantizationConfig(weight=OCP_MXFP8_STATIC_SPEC, input_tensors=OCP_MXFP8_DYNAMIC_SPEC)

W_INT8_A_INT8_PER_TOKEN_DYNAMIC_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TOKEN_DYNAMIC_SPEC, weight=INT8_PER_CHANNEL_SPEC
)

W_BFP16_CONFIG = QuantizationConfig(weight=W_BFP16_SPEC)
W_BFP16_A_BFP16_CONFIG = QuantizationConfig(input_tensors=A_BFP16_SPEC, weight=W_BFP16_SPEC)

W_MX6_CONFIG = QuantizationConfig(weight=W_MX6_SPEC)
W_MX6_A_MX6_CONFIG = QuantizationConfig(input_tensors=A_MX6_SPEC, weight=W_MX6_SPEC)

# quant_scheme for testing, not applied on the specific backend
W_UINT4_A_UINT4_PER_CHANNEL = QuantizationConfig(
    input_tensors=UINT4_PER_CHANNEL_ASYM_SPEC, weight=UINT4_PER_CHANNEL_ASYM_SPEC
)
W_UINT4_PER_TOKEN_A_INT8_PER_CHANNEL = QuantizationConfig(
    input_tensors=INT8_PER_TOKEN_DYNAMIC_SPEC, weight=UINT4_PER_CHANNEL_ASYM_SPEC
)
W_UINT4_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=UINT4_PER_CHANNEL_SYM_SPEC
)
W_UINT4_PER_GROUP_A_INT8_PER_TENSOR_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=UINT4_PER_GROUP_SYM_SPEC
)
W_FP8_A_FP8_O_FP8_PER_CHANNEL_SYM_CONFIG = QuantizationConfig(
    input_tensors=FP8_PER_TENSOR_SPEC, output_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_CHANNEL_SPEC
)
W_INT8_A_INT8_PER_TENSOR_PERCENTILE_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_PERCENTILE_SPEC
)
W_INT8_A_INT8_PER_TENSOR_MSE_CONFIG = QuantizationConfig(
    input_tensors=INT8_PER_TENSOR_SPEC, weight=INT8_PER_TENSOR_MSE_SPEC
)
W_UINT8_A_UINT8_PER_TENSOR_PERCENTILE_CONFIG = QuantizationConfig(
    input_tensors=UINT8_PER_TENSOR_SPEC, weight=UINT8_PER_TENSOR_PERCENTILE_SPEC
)
W_UINT8_A_UINT8_PER_TENSOR_MSE_CONFIG = QuantizationConfig(
    input_tensors=UINT8_PER_TENSOR_SPEC, weight=UINT8_PER_TENSOR_MSE_SPEC
)
W_UINT4_PER_CHANNEL_SYM_CONFIG = QuantizationConfig(weight=UINT4_PER_CHANNEL_SYM_SPEC)
W_UINT4_PER_CHANNEL_ASYM_CONFIG = QuantizationConfig(weight=UINT4_PER_CHANNEL_ASYM_SPEC)
W_UINT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_SYM_SPEC)
W_INT4_PER_CHANNEL_ASYM_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_ASYM_SPEC)
W_INT4_PER_GROUP_ASYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_ASYM_SPEC)
W_MXINT8_CONFIG = QuantizationConfig(weight=ocp_mxint8_spec(is_dynamic=False))
W_BFP16_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=BFP16_PER_GROUP_SYM_SPEC)
W_MXFP4_DIFFS_SYM_CONFIG = QuantizationConfig(weight=OCP_MX_SEPERATED_FP4_PER_GROUP_DIFFS_SPEC)

# Two stage config
W_FP4_SCALE_FP8_CONFIG = QuantizationConfig(weight=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC)
W_FP4_A_FP4_SCALE_FP8_CONFIG = QuantizationConfig(
    input_tensors=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC_DYNAMIC, weight=FP4_PER_GROUP_FP8_PER_TENSOR_SCALE_SPEC
)

INT4_PER_CHANNEL_SPEC = Int4PerChannelSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=0, is_dynamic=False
).to_quantization_spec()
INT4_PER_GROUP_SYM_SPEC = Int4PerGroupSpec(
    symmetric=True, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()
INT2_PER_GROUP_ASYM_SPEC = Int2PerGroupSpec(
    symmetric=False, scale_type="float", round_method="half_even", ch_axis=1, is_dynamic=False, group_size=128
).to_quantization_spec()

W_FP8_A_FP8_PER_TENSOR_CONFIG = QuantizationConfig(input_tensors=FP8_PER_TENSOR_SPEC, weight=FP8_PER_TENSOR_SPEC)
W_INT4_PER_CHANNEL_CONFIG = QuantizationConfig(weight=INT4_PER_CHANNEL_SPEC)
W_INT4_PER_GROUP_SYM_CONFIG = QuantizationConfig(weight=INT4_PER_GROUP_SYM_SPEC)
W_UINT4_PER_GROUP_CONFIG = QuantizationConfig(weight=UINT4_PER_GROUP_ASYM_SPEC)
W_INT2_PER_GROUP_CONFIG = QuantizationConfig(weight=INT2_PER_GROUP_ASYM_SPEC)


QUANT_SCHEME_TO_CONFIG = {
    "w_bfp16": W_BFP16_CONFIG,
    "w_bfp16_a_bfp16": W_BFP16_A_BFP16_CONFIG,
    "w_bfp16_per_group_sym": W_BFP16_PER_GROUP_SYM_CONFIG,
    "w_fp4_a_fp4_scale_fp8": W_FP4_A_FP4_SCALE_FP8_CONFIG,
    "w_fp4_scale_fp8": W_FP4_SCALE_FP8_CONFIG,
    "w_fp8_a_fp8": W_FP8_A_FP8_PER_TENSOR_CONFIG,
    "w_fp8_a_fp8_o_fp8": W_FP8_A_FP8_OFP8_PER_TENSOR_CONFIG,
    "w_fp8_per_channel_sym": W_FP8_A_FP8_O_FP8_PER_CHANNEL_SYM_CONFIG,
    "w_int4_per_channel_asym": W_INT4_PER_CHANNEL_ASYM_CONFIG,
    "w_int4_per_channel_sym": W_INT4_PER_CHANNEL_CONFIG,
    "w_int8_a_int8_per_tensor_sym": W_INT8_A_INT8_PER_TENSOR_CONFIG,
    "w_int8_a_int8_per_tensor_sym_dynamic": W_INT8_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
    # only support batch_size=1 for activation per-token
    "w_int8_a_int8_per_token_dynamic": W_INT8_A_INT8_PER_TOKEN_DYNAMIC_CONFIG,
    "w_int8_per_channel_a_int8_per_tensor_sym": W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG,
    "w_int8_per_channel_a_int8_per_tensor_sym_dynamic": W_INT8_PER_CHANNEL_A_INT8_PER_TENSOR_DYNAMIC_CONFIG,
    "w_int8_per_group_sym": W_INT8_PER_GROUP_CONFIG,
    "w_int8_per_tensor_mse": W_INT8_A_INT8_PER_TENSOR_MSE_CONFIG,
    "w_int8_per_tensor_percentile": W_INT8_A_INT8_PER_TENSOR_PERCENTILE_CONFIG,
    "w_int8_per_tensor_sym": W_INT8_PER_TENSOR_CONFIG,
    "w_mx6": W_MX6_CONFIG,
    "w_mx6_a_mx6": W_MX6_A_MX6_CONFIG,
    "w_mxfp4_diffs": W_MXFP4_DIFFS_SYM_CONFIG,
    "w_mxfp8": W_MXFP8_CONFIG,
    "w_mxfp8_a_mxfp8": W_MXFP8_A_MXFP8_CONFIG,
    "w_mxint8": W_MXINT8_CONFIG,
    "w_uint4_a_uint4_per_channel": W_UINT4_A_UINT4_PER_CHANNEL,
    "w_uint4_per_channel_a_int8_per_tensor": W_UINT4_PER_CHANNEL_A_INT8_PER_TENSOR_CONFIG,
    "w_uint4_per_channel_asym": W_UINT4_PER_CHANNEL_ASYM_CONFIG,
    "w_uint4_per_channel_sym": W_UINT4_PER_CHANNEL_SYM_CONFIG,
    "w_uint4_per_group_a_int8_per_tensor": W_UINT4_PER_GROUP_A_INT8_PER_TENSOR_CONFIG,
    "w_uint4_per_token_a_int8_per_channel": W_UINT4_PER_TOKEN_A_INT8_PER_CHANNEL,
    "w_uint8_a_uint8_per_tensor_asym": W_UINT8_A_UINT8_PER_TENSOR_CONFIG,
    "w_uint8_per_group_asym": W_UINT8_PER_GROUP_CONFIG,
    "w_uint8_per_tensor_mse": W_UINT8_A_UINT8_PER_TENSOR_MSE_CONFIG,
    "w_uint8_per_tensor_percentile": W_UINT8_A_UINT8_PER_TENSOR_PERCENTILE_CONFIG,
}


def get_global_config(
    quant_scheme: str, group_size: int, scale_format: str = "e4m3", scale_calculation_mode: str = "even"
) -> QuantizationConfig:
    if quant_scheme not in SUPPORTED_QUANT_SCHEME:
        raise ValueError(
            f"The quant_scheme {quant_scheme} is not supported, only {SUPPORTED_QUANT_SCHEME} are supported."
        )

    if quant_scheme in OCP_MX_QUANT_SCHEME and group_size != 32:
        raise ValueError(
            f"The quant_scheme {quant_scheme} requires group_size=32, got group_size={group_size}. Please use the option `--group-size 32`."
        )

    if quant_scheme in QUANT_SCHEME_TO_CONFIG:
        global_quant_config = QUANT_SCHEME_TO_CONFIG[quant_scheme]
    elif quant_scheme == "w_uint4_a_bfloat16_per_group_asym":
        global_quant_config = W_UINT4_A_BFLOAT16_PER_GROUP_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == "w_mxfp6_e3m2":
        global_quant_config = QuantizationConfig(weight=ocp_mxfp6_e3m2_spec(scale_calculation_mode, False))
    elif quant_scheme == "w_mxfp6_e2m3":
        global_quant_config = QuantizationConfig(weight=ocp_mxfp6_e2m3_spec(scale_calculation_mode, False))
    elif quant_scheme == "w_mxfp4_a_fp8_per_group":
        global_quant_config = QuantizationConfig(
            input_tensors=FP8_PER_TENSOR_SPEC, weight=ocp_mxfp4_spec(scale_calculation_mode, False)
        )
    elif quant_scheme == "w_mxfp4_a_mxfp4":
        global_quant_config = QuantizationConfig(
            input_tensors=fp4_per_group_sym_spec(group_size, "e8m0", scale_calculation_mode, True),
            weight=fp4_per_group_sym_spec(group_size, "e8m0", scale_calculation_mode, False),
        )
    elif quant_scheme == "w_fp4_a_fp4_dynamic_per_group_sym_gs16_scale_e8m0":
        global_quant_config = QuantizationConfig(
            input_tensors=FP4_PER_GROUP_SYM_DYNAMIC_GS16_SPEC, weight=FP4_PER_GROUP_SYM_STATIC_GS16_SPEC
        )
    elif quant_scheme == "w_mxfp4_a_mxfp6":
        global_quant_config = QuantizationConfig(
            input_tensors=ocp_mxfp6_e2m3_spec(scale_calculation_mode, True),
            weight=ocp_mxfp4_spec(scale_calculation_mode, False),
        )
    elif quant_scheme == "w_int4_per_group_asym":
        global_quant_config = W_INT4_PER_GROUP_ASYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == "w_uint4_per_group_sym":
        global_quant_config = W_UINT4_PER_GROUP_SYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == "w_mxfp4_a_mxfp8":
        global_quant_config = QuantizationConfig(
            input_tensors=ocp_mxfp4_spec(scale_calculation_mode, True),
            weight=ocp_mxfp4_spec(scale_calculation_mode, False),
        )
    elif quant_scheme == "w_mxfp4":
        global_quant_config = QuantizationConfig(weight=ocp_mxfp4_spec(scale_calculation_mode, False))
    elif quant_scheme == "w_fp4_per_group":
        global_quant_config = QuantizationConfig(weight=fp4_per_group_sym_spec(group_size, scale_format, None, False))
    elif quant_scheme == "w_mxfp6_e2m3_a_mxfp6_e2m3":
        global_quant_config = QuantizationConfig(
            input_tensors=ocp_mxfp6_e2m3_spec(scale_calculation_mode, True),
            weight=ocp_mxfp6_e2m3_spec(scale_calculation_mode, False),
        )
    elif quant_scheme == "w_mxfp6_e3m2_a_mxfp6_e3m2":
        global_quant_config = QuantizationConfig(
            input_tensors=ocp_mxfp6_e3m2_spec(scale_calculation_mode, True),
            weight=ocp_mxfp6_e3m2_spec(scale_calculation_mode, False),
        )
    elif quant_scheme == "w_fp6_e2m3_per_group_a_fp6_e2m3_per_group":
        global_quant_config = QuantizationConfig(
            input_tensors=fp6_e2m3_per_group_sym_spec(group_size, scale_format, None, True),
            weight=fp6_e2m3_per_group_sym_spec(group_size, scale_format, None, False),
        )
    elif quant_scheme == "w_fp6_e3m2_per_group_a_fp6_e3m2_per_group":
        global_quant_config = QuantizationConfig(
            input_tensors=fp6_e3m2_per_group_sym_spec(group_size, scale_format, None, True),
            weight=fp6_e3m2_per_group_sym_spec(group_size, scale_format, None, False),
        )
    elif quant_scheme == "w_int4_per_group_sym":
        global_quant_config = W_INT4_PER_GROUP_SYM_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == "w_uint4_per_group_asym":
        global_quant_config = W_UINT4_PER_GROUP_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    elif quant_scheme == "w_int2_per_group_asym":
        global_quant_config = W_INT2_PER_GROUP_CONFIG
        global_quant_config.weight.set_group_size(group_size)
    else:
        raise ValueError(f"please set global quant config for {quant_scheme} at customized_configuration.py")

    return global_quant_config
