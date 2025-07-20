"""
Copyright (C) 2023, Advanced Micro Devices, Inc. All rights reserved.
SPDX-License-Identifier: MIT

Implements the Brevitas quantization config and quantizer shim based on Quark interface.
This file should live outside of Quark codebase.
"""

import dataclasses
import torch
import enum
from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader
from diffusers.models import attention_processor
from quark import torch as quark_torch
from quark.shares.utils import log

from brevitas.graph import calibrate, equalize, quantize, base, gptq
from brevitas.utils.torch_utils import KwargsForwardHook
from brevitas.core.stats.stats_op import NegativeMinOrZero
from brevitas.inject.enum import StatsOp
from brevitas_examples.common.generative.quantize import (
    generate_quantizers,
    generate_quant_maps,
)
from brevitas_examples.stable_diffusion.sd_quant import nn as sd_qnn
from brevitas_examples.imagenet_classification.ptq import ptq_common
from dependencies import value

from typing import List, Union, Dict, Optional, Any


logger = log.ScreenLogger(__name__)


class BrevitasQuantizationMode(enum.Enum):
    FX = "fx"
    LAYERWISE = "layerwise"


@dataclasses.dataclass
class BrevitasGraphModeQuantizationConfig:
    scale_factor_type: str
    bias_bit_width: Optional[int]
    weight_bit_width: int
    weight_narrow_range: bool
    weight_param_method: str
    weight_quant_granularity: str
    weight_quant_type: str
    layerwise_first_last_bit_width: int
    act_bit_width: int
    act_param_method: str
    act_quant_percentile: float
    act_quant_type: str
    quant_format: str
    layerwise_first_last_mantissa_bit_width: int
    layerwise_first_last_exponent_bit_width: int
    weight_mantissa_bit_width: int
    weight_exponent_bit_width: int
    act_mantissa_bit_width: int
    act_exponent_bit_width: int
    act_scale_computation_type: str
    uint_sym_act_for_unsigned_values: bool


@dataclasses.dataclass
class BrevitasImageClassificationQuantizationConfig:
    dtype: torch.dtype
    device: str
    # Preprocess
    graph_eq_iterations: int
    graph_eq_merge_bias: bool
    merge_bn: bool
    channel_splitting_ratio: float
    channel_splitting_split_input: bool
    # Quantize
    quantize_config: BrevitasGraphModeQuantizationConfig
    gptq: bool

    # bias_correction
    bias_correction: bool

    # activation_equalization
    activation_equalization: bool
    activation_equalization_alpha: Optional[float] = None
    activation_equalization_exclude_blacklist: bool = False

    mode: BrevitasQuantizationMode = BrevitasQuantizationMode.FX


@dataclasses.dataclass
class BrevitasSDXLQuantizationConfig:
    dtype: torch.dtype
    device: str
    weight_bit_width: int
    weight_quant_format: str
    weight_quant_type: str
    weight_param_method: str
    weight_scale_precision: str
    weight_quant_granularity: str
    weight_group_size: int
    quantize_weight_zero_point: bool
    quantize_input_zero_point: bool
    input_bit_width: int
    input_quant_format: str
    input_scale_type: str
    input_scale_precision: str
    input_param_method: str
    input_quant_type: str
    input_quant_granularity: str
    input_scale_stats_op: str
    input_zp_stats_op: str
    use_ocp: bool
    use_fnuz: bool

    linear_input_bit_width: int
    linear_weight_bit_width: int

    conv_input_bit_width: int
    conv_weight_bit_width: int

    quantize_sdp_1: bool
    quantize_sdp_2: bool

    bias_correction: bool

    gptq: bool

    activation_equalization: bool
    activation_equalization_alpha: Optional[float] = None
    activation_equalization_exclude_blacklist: bool = False

    blacklist: Optional[List[str]] = None

    mode: BrevitasQuantizationMode = BrevitasQuantizationMode.LAYERWISE


class BrevitasModelQuantizer(quark_torch.ModelQuantizer):
    config: Union[
        BrevitasSDXLQuantizationConfig, BrevitasImageClassificationQuantizationConfig
    ]

    def init_config(self) -> None:
        # TODO: brevitas config validation
        pass

    def quantize_model(
        self,
        model: nn.Module,
        dataloader: Optional[
            Union[
                DataLoader[torch.Tensor],
                DataLoader[List[Dict[str, torch.Tensor]]],
                DataLoader[Dict[str, torch.Tensor]],
            ]
        ] = None,
    ) -> nn.Module:
        # Step1[optional]: Pre quant optimization
        model = self._apply_pre_quantization_optimization(model, dataloader)

        # Step2: Prepare quantization model for graph mode and eager mode
        model = self._prepare_model(model)

        # Step3[optional]: Do calibration
        model = self._do_calibration(model, dataloader)

        # NB: Brevitas applies advanced algorithm after calibration.
        # Step4[optional]: Apply Advanced quant algo such as gptq/awq ...
        model = self._apply_advanced_quant_algo(model, dataloader)

        return model

    def _apply_pre_quantization_optimization(
        self, model: nn.Module, dataloader: Optional[Any] = None
    ) -> nn.Module:
        if self.config.mode == BrevitasQuantizationMode.FX:
            model = quantize.preprocess_for_quantize(
                model,
                trace_model=False,  # trace outside
                equalize_iters=self.config.graph_eq_iterations,
                equalize_merge_bias=self.config.graph_eq_merge_bias,
                merge_bn=self.config.merge_bn,
                channel_splitting_ratio=self.config.channel_splitting_ratio,
                channel_splitting_split_input=self.config.channel_splitting_split_input,
            )
        # NOTE: equalization is applied before calibration for brevitas sdxl.
        # Need to update api to pass dataloader.
        if self.config.activation_equalization:
            logger.info("Applying activation equalization")
            with equalize.activation_equalization_mode(
                model,
                alpha=self.config.activation_equalization_alpha,
                layerwise=self.config.mode == BrevitasQuantizationMode.LAYERWISE,
                blacklist_layers=(
                    self.config.blacklist
                    if self.config.activation_equalization_exclude_blacklist
                    else None
                ),
                add_mul_node=True,
            ):
                # Workaround to expose `in_features` attribute from the Hook Wrapper
                for m in model.modules():
                    if isinstance(m, KwargsForwardHook) and hasattr(
                        m.module, "in_features"
                    ):
                        m.in_features = m.module.in_features
                for data in tqdm(dataloader):
                    model.calibration_callable(data)
            logger.info("Activation equalization done")

            # Workaround to expose `in_features` attribute from the EqualizedModule Wrapper
            for m in model.modules():
                if isinstance(m, equalize.EqualizedModule) and hasattr(
                    m.layer, "in_features"
                ):
                    m.in_features = m.layer.in_features
        return model

    def _prepare_model(self, model: nn.Module) -> nn.Module:
        if self.config.mode is BrevitasQuantizationMode.FX:
            return self._prepare_model_fx(model)
        elif self.config.mode is BrevitasQuantizationMode.LAYERWISE:
            return self._prepare_model_layer_wise(model)
        raise ValueError(f"Unsupported mode: {self.config.mode}")

    def _prepare_model_fx(self, model: nn.Module) -> nn.Module:
        return ptq_common.quantize_model(
            model,
            dtype=self.config.dtype,
            device=self.config.device,
            backend="fx",
            **dataclasses.asdict(self.config.quantize_config),
        )

    def _prepare_model_layer_wise(self, model: nn.Module) -> nn.Module:
        input_kwargs = dict()
        if self.config.input_scale_stats_op == "minmax":

            @value
            def input_scale_stats_type():
                if self.config.input_quant_type == "asym":
                    input_scaling_stats_op = StatsOp.MIN_MAX
                else:
                    input_scaling_stats_op = StatsOp.MAX
                return input_scaling_stats_op

            input_kwargs["scaling_stats_op"] = input_scale_stats_type

        if self.config.input_zp_stats_op == "minmax":

            @value
            def input_zp_stats_type():
                if self.config.input_quant_type == "asym":
                    zero_point_stats_impl = NegativeMinOrZero
                    return zero_point_stats_impl

            input_kwargs["zero_point_stats_impl"] = input_zp_stats_type

        quantizers = generate_quantizers(
            dtype=self.config.dtype,
            device=self.config.device,
            weight_bit_width=self.config.weight_bit_width,
            weight_quant_format=self.config.weight_quant_format,
            weight_quant_type=self.config.weight_quant_type,
            weight_param_method=self.config.weight_param_method,
            weight_scale_precision=self.config.weight_scale_precision,
            weight_quant_granularity=self.config.weight_quant_granularity,
            weight_group_size=self.config.weight_group_size,
            quantize_weight_zero_point=self.config.quantize_weight_zero_point,
            quantize_input_zero_point=self.config.quantize_input_zero_point,
            input_bit_width=self.config.input_bit_width,
            input_quant_format=self.config.input_quant_format,
            input_scale_type=self.config.input_scale_type,
            input_scale_precision=self.config.input_scale_precision,
            input_param_method=self.config.input_param_method,
            input_quant_type=self.config.input_quant_type,
            input_quant_granularity=self.config.input_quant_granularity,
            use_ocp=self.config.use_ocp,
            use_fnuz=self.config.use_fnuz,
            input_kwargs=input_kwargs,
        )

        layer_map = generate_quant_maps(
            *quantizers,
            self.config.dtype,
            self.config.device,
            self.config.input_quant_format,
            False,
        )

        linear_qkwargs = layer_map[torch.nn.Linear][1]
        linear_qkwargs["input_quant"] = (
            None
            if self.config.linear_input_bit_width == 0
            else linear_qkwargs["input_quant"]
        )
        linear_qkwargs["weight_quant"] = (
            None
            if self.config.linear_weight_bit_width == 0
            else linear_qkwargs["weight_quant"]
        )
        layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], linear_qkwargs)

        conv_qkwargs = layer_map[torch.nn.Conv2d][1]
        conv_qkwargs["input_quant"] = (
            None
            if self.config.conv_input_bit_width == 0
            else conv_qkwargs["input_quant"]
        )
        conv_qkwargs["weight_quant"] = (
            None
            if self.config.conv_weight_bit_width == 0
            else conv_qkwargs["weight_quant"]
        )
        layer_map[torch.nn.Conv2d] = (layer_map[torch.nn.Conv2d][0], conv_qkwargs)

        if self.config.quantize_sdp_1 or self.config.quantize_sdp_2:
            float_sdpa_quantizers = generate_quantizers(
                dtype=self.config.dtype,
                device=self.config.device,
                weight_bit_width=self.config.weight_bit_width,
                weight_quant_format="e4m3",
                weight_quant_type="sym",
                weight_param_method=self.config.weight_param_method,
                weight_scale_precision=self.config.weight_scale_precision,
                weight_quant_granularity=self.config.weight_quant_granularity,
                weight_group_size=self.config.weight_group_size,
                quantize_weight_zero_point=self.config.quantize_weight_zero_point,
                quantize_input_zero_point=self.config.quantize_input_zero_point,
                input_bit_width=self.config.input_bit_width,
                input_quant_format="e4m3",
                input_scale_type=self.config.input_scale_type,
                input_scale_precision=self.config.input_scale_precision,
                input_param_method=self.config.input_param_method,
                input_quant_type="sym",
                input_quant_granularity=self.config.input_quant_granularity,
                use_ocp=self.config.use_ocp,
                use_fnuz=self.config.use_fnuz,
                input_kwargs=input_kwargs,
            )
            # We generate all quantizers, but we are only interested in activation quantization for
            # the output of softmax and the output of QKV
            input_quant = float_sdpa_quantizers[0]
            input_quant = input_quant.let(
                **{"bit_width": self.config.linear_output_bit_width}
            )
            if self.config.quantize_sdp_2:
                rewriter = base.ModuleToModuleByClass(
                    attention_processor.Attention,
                    sd_qnn.QuantAttention,
                    softmax_output_quant=input_quant,
                    query_dim=lambda module: module.to_q.in_features,
                    dim_head=lambda module: int(1 / (module.scale**2)),
                    processor=attention_processor.AttnProcessor(),
                    is_equalized=self.config.activation_equalization,
                )
                import brevitas.config as config

                config.IGNORE_MISSING_KEYS = True
                model = rewriter.apply(model)
                config.IGNORE_MISSING_KEYS = False
                model = model.to(self.config.device)
                model = model.to(self.config.dtype)
            quant_kwargs = layer_map[torch.nn.Linear][1]
            what_to_quantize = []
            if self.config.quantize_sdp_1:
                what_to_quantize.extend(["to_q", "to_k"])
            if self.config.quantize_sdp_2:
                what_to_quantize.extend(["to_v"])
            quant_kwargs["output_quant"] = lambda module, name: (
                input_quant
                if any(ending in name for ending in what_to_quantize)
                else None
            )
            layer_map[torch.nn.Linear] = (layer_map[torch.nn.Linear][0], quant_kwargs)

        print(f"layer_map:{layer_map}")

        model = quantize.layerwise_quantize(
            model,
            compute_layer_map=layer_map,
            name_blacklist=self.config.blacklist,
        )

        return model

    def _do_calibration(
        self,
        model: nn.Module,
        dataloader: Optional[Any] = None,
    ) -> nn.Module:
        if (
            isinstance(self.config, BrevitasImageClassificationQuantizationConfig)
            and self.config.quantize_config.act_scale_computation_type != "static"
        ):
            return model
        with torch.no_grad(), calibrate.calibration_mode(model):
            for data in tqdm(dataloader):
                model.calibration_callable(data)
        return model

    def _apply_advanced_quant_algo(
        self,
        model: nn.Module,
        dataloader: Optional[
            Union[
                DataLoader[torch.Tensor],
                DataLoader[List[Dict[str, torch.Tensor]]],
                DataLoader[Dict[str, torch.Tensor]],
            ]
        ] = None,
    ) -> nn.Module:
        if self.config.gptq:
            logger.info("Applying GPTQ. It can take several hours")
            with torch.no_grad(), gptq.gptq_mode(
                model,
                create_weight_orig=False,
                use_quant_activations=False,
                return_forward_output=True,
                act_order=True,
            ) as gptq_ctx:
                for _ in tqdm(range(gptq_ctx.num_layers)):
                    for data in tqdm(dataloader):
                        model.calibration_callable(data)
                    gptq_ctx.update()
                    torch.cuda.empty_cache()
            logger.info("GPTQ done")

        if self.config.bias_correction:
            logger.info("Applying bias correction")
            with torch.no_grad(), calibrate.bias_correction_mode(model):
                for data in tqdm(dataloader):
                    model.calibration_callable(data)
            logger.info("Bias correction done")
        return model
