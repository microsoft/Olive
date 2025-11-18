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
from olive.common.utils import StrEnumBase, get_attr
from olive.constants import PrecisionBits
from olive.model import HfModelHandler
from olive.passes import Pass
from olive.passes.pass_config import BasePassConfig, PassConfigParam
from olive.passes.pytorch.train_utils import get_calibration_dataset, kl_div_loss, load_hf_base_model

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec


logger = logging.getLogger(__name__)


class SelectiveMixedPrecision(Pass):
    """Annotate the model with mixed precision information.

    This pass is used to annotate the model with mixed precision information, which can be used by other passes
    to quantize the model. The pass will add a model attribute `mixed_precision_info` that contains
    information about the default precision along with overrides for specific layers.

    The supported algorithms are:
    - Layer id based heuristic:
        - k_quant_last: LM head in high precision.
        - k_quant_down: LM head + Down projection from first 1/8 and last 1/8 layers, and every 3rd layer in between in high precision.
        - k_quant_mixed: LM head + QKV and Down projection from first 1/8 and last 1/8 layers, and every 3rd layer in between in high precision.
    - Sensitivity score based:
        - snr: Signal-to-Noise Ratio based selection.
        - snr_relative: Relative SNR (between low and high precision) based selection.
        - iqe: Inverse of Integer Quantization Error based selection.
        - iqe_relative: Relative IQE (between low and high precision) based selection.
        - kld_gradient: KL Divergence gradient based selection.
    """

    class Algorithm(StrEnumBase):
        """The algorithm to use for mixed precision."""

        IQE = "iqe"
        IQE_RELATIVE = "iqe_relative"
        K_QUANT_DOWN = "k_quant_down"
        K_QUANT_MIXED = "k_quant_mixed"
        K_QUANT_LAST = "k_quant_last"
        KLD_GRADIENT = "kld_gradient"
        SNR = "snr"
        SNR_RELATIVE = "snr_relative"

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
                description="The default group size. Only used for snr, iqe and kld_gradient algorithms.",
            ),
            "sym": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Whether to use symmetric quantization by default. Only used for snr, iqe and kld_gradient"
                    " algorithms."
                ),
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
                    "The group size for high precision layers. Only used for snr, iqe and kld_gradient algorithms. If"
                    " None, use group_size."
                ),
            ),
            "high_sym": PassConfigParam(
                type_=bool,
                default_value=None,
                description=(
                    "Whether to use symmetric quantization for high precision layers. Only used for snr, iqe and"
                    " kld_gradient algorithms. If None, use sym."
                ),
            ),
            "ratio": PassConfigParam(
                type_=float,
                default_value=None,
                description=(
                    "The ratio of default precision parameters to total parameters. Only used for snr, iqe and"
                    " kld_gradient algorithms. Must be provided when using these algorithms."
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

        # clear cached model
        model.model = None
        model_wrapper = ModelWrapper.from_model(load_hf_base_model(model))

        if config.algorithm.startswith("k_quant"):
            default, overrides = self.get_k_quant_config(model_wrapper, config.algorithm, config.bits, config.high_bits)
        else:
            default, overrides = self.get_scored_config(
                model,
                model_wrapper,
                config.algorithm,
                config.bits,
                config.group_size,
                config.sym,
                config.high_bits,
                config.high_group_size if config.high_group_size is not None else config.group_size,
                config.high_sym if config.high_sym is not None else config.sym,
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
        """Get mixed precision config for k-quant algorithms."""
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
    def get_scored_config(
        handler: HfModelHandler,
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
        """Get mixed precision config based on sensitivity scores."""
        quantizer = WeightQuantizer(bits=bits, group_size=group_size, symmetric=symmetric)
        high_quantizer = WeightQuantizer(
            bits=high_bits,
            group_size=high_group_size,
            symmetric=high_symmetric,
        )
        device = "cuda" if torch.cuda.is_available() else "cpu"

        algo_func = (
            SelectiveMixedPrecision.get_kld_scores
            if algorithm == SelectiveMixedPrecision.Algorithm.KLD_GRADIENT
            else SelectiveMixedPrecision.get_snr_iqe_scores
        )
        module_numels, module_scores = algo_func(
            handler,
            model_wrapper.model,
            algorithm,
            quantizer,
            high_quantizer,
            device,
        )

        threshold = sum(module_numels.values()) * (1 - ratio)
        # ascending order, lower score means more sensitive and should be in higher precision
        sorted_modules = sorted(module_scores, key=lambda item: module_scores[item], reverse=False)
        overrides = {}
        high_override_config = {"bits": high_bits, "group_size": high_group_size, "symmetric": high_symmetric}
        total = 0
        for module_name in sorted_modules:
            total += module_numels[module_name]
            overrides[module_name] = high_override_config
            if total >= threshold:
                break
        logger.info(
            "Selected %d modules for high precision out of %d modules. Ratio of low precision: %.4f",
            len(overrides),
            len(module_numels),
            1 - total / sum(module_numels.values()),
        )

        return {"bits": bits, "group_size": group_size, "symmetric": symmetric}, overrides

    @staticmethod
    def get_snr_iqe_scores(
        handler: HfModelHandler,
        model: torch.nn.Module,
        algorithm: SelectiveMixedPrecision.Algorithm,
        quantizer: WeightQuantizer,
        high_quantizer: WeightQuantizer,
        device: str,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Compute SNR or IQE based sensitivity scores."""
        score_func = (
            SelectiveMixedPrecision.compute_snr if algorithm.startswith("snr") else SelectiveMixedPrecision.compute_iqe
        )

        module_numels = {}
        module_scores = {}

        def should_include(module, _):
            return isinstance(module, torch.nn.Linear)

        @torch.no_grad()
        def process_module(module, module_name):
            module_numels[module_name] = module.weight.numel()
            module.to(device)

            score = score_func(module.weight, quantizer.fake_quantize(module.weight))
            if algorithm.endswith("_relative"):
                high_score = score_func(module.weight, high_quantizer.fake_quantize(module.weight))
                if algorithm.startswith("snr"):
                    # SNR is in dB, so we subtract
                    score -= high_score
                else:
                    # IQE is inverted, so we divide
                    score /= high_score
            module_scores[module_name] = score
            return module.cpu()

        replace_matching_submodules(model, should_include, process_module, description="Computing SNR/IQE scores")
        return module_numels, module_scores

    @staticmethod
    def get_kld_scores(
        handler: HfModelHandler,
        model: torch.nn.Module,
        algorithm: SelectiveMixedPrecision.Algorithm,
        quantizer: WeightQuantizer,
        high_quantizer: WeightQuantizer,
        device: str,
    ) -> tuple[dict[str, int], dict[str, float]]:
        """Compute KL Divergence gradient based sensitivity scores.

        The gradients are computed using a calibration dataset and the KL Divergence loss between
        the outputs of the original model and the fake quantized model.
        """
        # based on mlx-lm implementation at https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/quant/dynamic_quant.py
        from tqdm.auto import tqdm

        # TODO(jambayk): make data_config configurable
        data = get_calibration_dataset(handler, max_seq_len=512, max_samples=256)

        model.to(device).eval()
        q_model = deepcopy(model).to(device).eval()

        # freeze all parameters
        for param in q_model.parameters():
            param.requires_grad = False
        # enable gradient checkpointing
        q_model.gradient_checkpointing_enable()

        # replace the weights in qmodel with low-bit quantized weights
        module_numels = {}
        q_layers: dict[str, torch.nn.Module] = {}
        grad_accum: dict[str, torch.Tensor] = {}

        def should_include(module, _):
            return isinstance(module, torch.nn.Linear)

        @torch.no_grad()
        def process_module(module, module_name):
            module_numels[module_name] = module.weight.numel()
            low_w = quantizer.fake_quantize(module.weight.data)
            module.weight.data = low_w
            module.weight.requires_grad = True
            q_layers[module_name] = module
            grad_accum[module_name] = torch.zeros_like(module.weight.data, dtype=torch.float32)
            return module

        replace_matching_submodules(
            q_model, should_include, process_module, description="Preparing for sensitivity estimation"
        )

        for batch in tqdm(data, desc="Estimating sensitivities"):
            inputs = {k: v.to(device) for k, v in batch.items()}

            with torch.no_grad():
                teacher_logits = model(**inputs).logits

            student_logits = q_model(**inputs).logits
            loss = kl_div_loss(student_logits, teacher_logits).mean()
            loss.backward()

            # accumulate gradients
            for name, layer in q_layers.items():
                grad_accum[name] += layer.weight.grad.data.detach().float()

            # zero grads
            q_model.zero_grad()

        @torch.no_grad()
        def compute_sensitivity(module_name: str) -> torch.Tensor:
            grad = grad_accum[module_name] / len(data)  # average gradient

            # high-precision quantization baseline
            high_w = high_quantizer.fake_quantize(get_attr(model, module_name).weight.data)

            # get sensitivity
            param_size_m = module_numels[module_name] / 1e6
            alignment = (grad * (q_layers[module_name].weight.data - high_w)).sum().item()
            return alignment / param_size_m

        # negative sensitivity because lower is more sensitive
        return module_numels, {name: -compute_sensitivity(name) for name in q_layers}

    @staticmethod
    def compute_snr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute signal-to-noise ratio in dB.

        Args:
            x (torch.Tensor): Original tensor.
            y (torch.Tensor): Quantized tensor.
            eps (float): Small value to avoid division by zero.

        Returns:
            float: SNR value in dB.

        """
        x = x.flatten().float()
        y = y.flatten().float()
        signal_norm = max(torch.norm(x).item(), eps)
        noise_norm = max(torch.norm(x - y).item(), eps)
        return 20 * math.log10(signal_norm / noise_norm)

    @staticmethod
    def compute_iqe(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
        """Compute the inverse of integer quantization error.

        Error is computed as max(mean((x - y)^2)) over the last dimension.

        Args:
            x (torch.Tensor): Original tensor.
            y (torch.Tensor): Quantized tensor.
            eps (float): Small value to avoid division by zero.

        Returns:
            float: Inverse of IQE score.

        """
        # based on nncf implementation at
        # https://github.com/openvinotoolkit/nncf/blob/develop/src/nncf/quantization/algorithms/weight_compression/weight_lowering.py
        iqe = torch.pow(x.float() - y.float(), 2).mean(-1).max().item()
        # invert so that lower score means more sensitive
        return 1 / (iqe + eps)
