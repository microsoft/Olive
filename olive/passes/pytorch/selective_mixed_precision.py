# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import TYPE_CHECKING

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import replace_matching_submodules, sort_layers_by_name
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import StrEnumBase
from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec


logger = logging.getLogger(__name__)


class SelectiveMixedPrecision(Pass):
    """Annotate the model with mixed precision information.

    This pass is used to annotate the model with mixed precision information, which can be used by other passes
    to quantize the model. The pass will add a model attribute `mixed_precision_info` that contains
    information about the default precision along with overrides for specific layers.
    """

    class Algorithm(StrEnumBase):
        """The algorithm to use for mixed precision."""

        K_QUANT_DOWN = "k_quant_down"
        K_QUANT_MIXED = "k_quant_mixed"
        K_QUANT_LAST = "k_quant_last"
        SNR = "snr"
        SNR_RELATIVE = "snr_relative"
        IQE = "iqe"
        IQE_RELATIVE = "iqe_relative"
        # TODO(jambayk): add other heuristic/algorithms

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "algorithm": PassConfigParam(
                type_=SelectiveMixedPrecision.Algorithm,
                required=True,
                description="The algorithm to use for mixed precision.",
            ),
            "bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS4,
                description="The default precision bits.",
            ),
            "group_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="The default group size. Only used for snr and iqe algorithms.",
            ),
            "symmetric": PassConfigParam(
                type_=bool,
                default_value=False,
                description="Whether to use symmetric quantization by default. Only used for snr and iqe algorithms.",
            ),
            "high_bits": PassConfigParam(
                type_=PrecisionBits,
                default_value=PrecisionBits.BITS8,
                description="The high precision bits for selected layers.",
            ),
            "high_group_size": PassConfigParam(
                type_=int,
                default_value=None,
                description=(
                    "The group size for high precision layers. Only used for snr and iqe algorithms. If None, use"
                    " group_size."
                ),
            ),
            "high_symmetric": PassConfigParam(
                type_=bool,
                default_value=None,
                description=(
                    "Whether to use symmetric quantization for high precision layers. Only used for snr and iqe"
                    " algorithms. If None, use symmetric."
                ),
            ),
            "ratio": PassConfigParam(
                type_=float,
                default_value=None,
                description=(
                    "The ratio of default precision parameters to total parameters. Only used for snr and iqe"
                    " algorithms. Must be provided when using these algorithms."
                ),
            ),
        }

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        if not config.algorithm.startswith("k_quant") and (config.ratio is None or not (0 < config.ratio < 1)):
            logger.error("When using %s algorithm, ratio must be provided and between 0 and 1.", config.algorithm)
            return False

        return True

    def _run_for_config(
        self, model: HfModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> HfModelHandler:
        """Run the selective mixed precision pass."""
        if not isinstance(model, HfModelHandler):
            raise ValueError("SelectiveMixedPrecision pass currently only supports HfModelHandler.")

        model_wrapper = ModelWrapper.from_model(model.load_model(cache_model=False))

        if config.algorithm.startswith("k_quant"):
            default, overrides = self.get_k_quant_config(model_wrapper, config.algorithm, config.bits, config.high_bits)
        else:
            default, overrides = self.get_snr_iqe_config(
                model_wrapper,
                config.algorithm,
                config.bits,
                config.group_size,
                config.symmetric,
                config.high_bits,
                config.high_group_size if config.high_group_size is not None else config.group_size,
                config.high_symmetric if config.high_symmetric is not None else config.symmetric,
                config.ratio,
            )

        lm_head_name = model_wrapper.get_lm_head()[1]
        if model_wrapper.config.tie_word_embeddings and lm_head_name in overrides:
            overrides[model_wrapper.get_embeds()[1][0]] = overrides[lm_head_name]

        # sort the overrides for better readability
        overrides = sort_layers_by_name(overrides)

        # Create output model with mixed precision info as model attribute
        # deepcopy is okay since loaded model is not cached
        output_model = deepcopy(model)
        output_model.model_attributes = output_model.model_attributes or {}
        output_model.model_attributes["mixed_precision_info"] = {
            "default": default,
            "overrides": overrides,
        }
        return output_model

    @staticmethod
    def get_k_quant_config(
        model_wrapper: ModelWrapper,
        algorithm: SelectiveMixedPrecision.Algorithm,
        bits: PrecisionBits,
        high_bits: PrecisionBits,
    ) -> tuple[dict, dict[str, dict]]:
        override_config = {"bits": high_bits}
        overrides = {model_wrapper.get_lm_head()[1]: override_config}

        if algorithm != SelectiveMixedPrecision.Algorithm.K_QUANT_LAST:
            layer_prefix = model_wrapper.get_layers()[1]
            num_layers = model_wrapper.num_hidden_layers
            for layer_idx, layer_wrapper in enumerate(model_wrapper.get_layer_wrappers()):
                if not (
                    layer_idx < num_layers / 8
                    or layer_idx >= 7 * num_layers / 8
                    or ((layer_idx - num_layers // 8) % 3 == 2)
                ):
                    continue

                # Add qkv
                if algorithm == SelectiveMixedPrecision.Algorithm.K_QUANT_MIXED:
                    for attn_input_name in layer_wrapper.get_attention_inputs(return_name=True)[1]:
                        overrides[f"{layer_prefix}.{layer_idx}.{attn_input_name}"] = override_config

                # Add down_proj
                for attn_output_name in layer_wrapper.get_mlp_outputs(return_name=True)[1]:
                    overrides[f"{layer_prefix}.{layer_idx}.{attn_output_name}"] = override_config

        return {"bits": bits}, overrides

    @staticmethod
    def get_snr_iqe_config(
        model_wrapper: ModelWrapper,
        algorithm: SelectiveMixedPrecision.Algorithm,
        bits: PrecisionBits,
        group_size: int,
        symmetric: bool,
        high_bits: PrecisionBits,
        high_group_size: int,
        high_symmetric: bool,
        ratio: float,
    ):
        quantizer = WeightQuantizer(bits=bits, group_size=group_size, symmetric=symmetric)
        score_func = (
            SelectiveMixedPrecision.compute_sqnr if algorithm.startswith("snr") else SelectiveMixedPrecision.compute_iqe
        )
        use_relative = algorithm.endswith("_relative")
        if use_relative:
            high_quantizer = WeightQuantizer(
                bits=high_bits,
                group_size=high_group_size,
                symmetric=high_symmetric,
            )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        module_numel = {}
        module_scores = {}

        def should_include(module, _):
            return isinstance(module, torch.nn.Linear)

        @torch.no_grad()
        def process_module(module, module_name):
            module_numel[module_name] = module.weight.numel()
            module.to(device)

            score = score_func(module.weight, quantizer.fake_quantize(module.weight))
            if use_relative:
                high_score = score_func(module.weight, high_quantizer.fake_quantize(module.weight))
                if algorithm.startswith("snr"):
                    # SNR is in dB, so we subtract
                    score -= high_score
                else:
                    # IQE is inverted, so we divide
                    score /= high_score
            module_scores[module_name] = score
            module.cpu()

        replace_matching_submodules(
            model_wrapper.model, should_include, process_module, description="Computing SNR/IQE scores"
        )

        threshold = sum(module_numel.values()) * (1 - ratio)
        # ascending order, lower score means more sensitive and should be in higher precision
        sorted_modules = sorted(module_scores, key=lambda item: module_scores[item], reverse=False)
        overrides = {}
        high_override_config = {"bits": high_bits, "group_size": high_group_size, "symmetric": high_symmetric}
        total = 0
        for module_name in sorted_modules:
            total += module_numel[module_name]
            overrides[module_name] = high_override_config
            if total >= threshold:
                break
        logger.info(
            "Selected %d modules for high precision out of %d modules. Ratio of low precision: %.4f",
            len(overrides),
            len(module_numel),
            1 - total / sum(module_numel.values()),
        )

        return {"bits": bits, "group_size": group_size, "symmetric": symmetric}, overrides

    @staticmethod
    def compute_sqnr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        x = x.flatten().float()
        y = y.flatten().float()
        signal_norm = max(torch.norm(x).item(), eps)
        noise_norm = max(torch.norm(x - y).item(), eps)
        return 20 * math.log10(signal_norm / noise_norm)

    @staticmethod
    def compute_iqe(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        iqe = torch.pow(x.float() - y.float(), 2).mean(-1).max().item()
        return 1 / (iqe + eps)
