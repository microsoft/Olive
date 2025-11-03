# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import (
    OliveHfQuantizationConfig,
    OliveHfQuantizationMethod,
    replace_matching_submodules,
    tie_quant_word_embeddings,
)
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.common.quant.utils import WeightQuantizer
from olive.constants import PrecisionBits
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import load_hf_base_model

if TYPE_CHECKING:
    from olive.model import HfModelHandler
    from olive.passes.pass_config import BasePassConfig


logger = logging.getLogger(__name__)


def get_quantizer_config(allow_embeds: bool = False) -> dict[str, PassConfigParam]:
    return {
        "bits": PassConfigParam(
            type_=PrecisionBits,
            default_value=PrecisionBits.BITS4,
            description="quantization bits. Default value is 4",
        ),
        "group_size": PassConfigParam(
            type_=int,
            default_value=128,
            description="Block size for quantization. Default value is 128.",
        ),
        "sym": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Symmetric quantization. Default value is False.",
        ),
        "lm_head": PassConfigParam(
            type_=bool,
            default_value=False,
            description="Whether to quantize the language model head. Default value is False.",
        ),
        **(
            {
                "embeds": PassConfigParam(
                    type_=bool,
                    default_value=False,
                    description="Whether to quantize the input embeddings. Default value is False.",
                )
            }
            if allow_embeds
            else {}
        ),
        "overrides": PassConfigParam(
            type_=dict,
            default_value=None,
            description=(
                "Optional dictionary to specify overrides for specific modules. The keys are module names and the"
                " values are dictionaries with any of the following keys: 'bits', 'symmetric', 'group_size'. If the"
                " same values are overridden by the model's mixed precision info, the mixed precision info takes"
                " precedence."
            ),
        ),
    }


def prepare_model(
    model: HfModelHandler, config: type[BasePassConfig], allow_embeds: bool = False, allow_quantized: bool = False
) -> tuple[ModelWrapper, OliveHfQuantizationConfig, bool]:
    """Prepare the model for quantization by adding quant_info to linear layers.

    Args:
        model: The HuggingFace model to prepare.
        config: Configuration object containing quantization parameters.
        allow_embeds: Whether to allow quantization of embedding layers.
        allow_quantized: Whether to allow already (partially) quantized models.

    Returns:
        A tuple containing ModelWrapper with prepared model, the quantization configuration, and a boolean indicating if the word embeddings are eligible for tieing.

    """
    if existing_qcfg := getattr(model.get_hf_model_config(), "quantization_config", None):
        if not allow_quantized:
            raise ValueError("Model is already quantized. Cannot quantize again using this pass.")
        if getattr(existing_qcfg, "quant_method", None) != OliveHfQuantizationMethod.OLIVE:
            raise ValueError("Model has an existing quantization configuration that is not compatible with this pass.")

    wrapper = ModelWrapper.from_model(load_hf_base_model(model, torch_dtype="auto"))
    wrapper.model.eval()
    wrapper.model.config.use_cache = False

    qcfg = get_quant_config(model, config)

    originally_tied_embeddings = wrapper.config.tie_word_embeddings
    if qcfg.lm_head or qcfg.embeds:
        wrapper.maybe_untie_word_embeddings()

    lm_head_name = wrapper.get_lm_head()[1]
    embeds_name = wrapper.get_embeds()[1]
    new_qargs: dict[str, dict[str, int | bool]] = {}

    def should_quantize(module: torch.nn.Module, name: str) -> bool:
        if isinstance(module, torch.nn.Linear):
            return name != lm_head_name or qcfg.lm_head
        if allow_embeds and isinstance(module, torch.nn.Embedding):
            return name == embeds_name
        return False

    def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
        # TODO(jambayk): validate that the module and config are compatible
        qargs = qcfg.get_qlinear_init_args(name)
        module.quant_info = QuantInfo(quantizer=WeightQuantizer(**qargs))
        new_qargs[name] = qargs
        return module

    replace_matching_submodules(
        wrapper.model,
        should_quantize,
        add_quant_info,
        description="Preparing model for quantization",
    )

    # merge the new_quant_settings into the existing quant_config
    if existing_qcfg:
        merged_qcfg_dict = existing_qcfg.to_dict()
        merged_qcfg_dict["overrides"] = existing_qcfg.overrides or {}
        for name, qargs in new_qargs.items():
            override = {k: v for k, v in qargs.items() if merged_qcfg_dict[k] != v}
            if override:
                merged_qcfg_dict["overrides"][name] = override
        merged_qcfg_dict["lm_head"] |= qcfg.lm_head
        merged_qcfg_dict["embeds"] |= qcfg.embeds
        qcfg = OliveHfQuantizationConfig(**merged_qcfg_dict)

    word_embeddings_eligible_for_tieing = (
        originally_tied_embeddings
        and embeds_name in new_qargs
        and lm_head_name in new_qargs
        and new_qargs[embeds_name] == new_qargs[lm_head_name]
    )

    return wrapper, qcfg, word_embeddings_eligible_for_tieing


def get_quant_config(model: HfModelHandler, config: type[BasePassConfig]) -> OliveHfQuantizationConfig:
    """Get quantization configuration with mixed precision support.

    Args:
        model: The HuggingFace model to get configuration for.
        config: Configuration object containing quantization parameters.

    Returns:
        OliveHfQuantizationConfig object with quantization settings.

    """
    quant_config = {
        "bits": config.bits,
        "symmetric": config.sym,
        "group_size": config.group_size,
        "lm_head": config.lm_head,
        "embeds": getattr(config, "embeds", False),
        "overrides": config.overrides or {},
    }
    if mp_info := (model.model_attributes or {}).get("mixed_precision_info"):
        for k, v in quant_config.items():
            if mp_info["default"].get(k) is not None and v != mp_info["default"][k]:
                logger.debug("Overriding %s with mixed precision info: %s", k, mp_info["default"][k])
                quant_config[k] = mp_info["default"][k]
        # merge overrides
        for name, override in mp_info.get("overrides", {}).items():
            merged = quant_config["overrides"].get(name, {}).copy()
            merged.update({k: v for k, v in override.items() if v is not None})
            quant_config["overrides"][name] = merged
    return OliveHfQuantizationConfig(**quant_config)


@dataclass
class QuantInfo:
    """Class to hold quantization information for GPTQ.

    This class stores all the necessary information for quantizing a layer,
    including the quantizer, computed scales and zero points, and calibration data.

    Attributes:
        quantizer: The weight quantizer used for quantization.
        scales: Computed scales for quantization. Set after processing.
        zero_points: Computed zero points for quantization. Set after processing.
        data: Calibration data including Hessian matrix and sample count.
              Format: {"H": torch.Tensor, "N": int} for gptq or None.

    """

    quantizer: WeightQuantizer
    scales: torch.Tensor | None = None
    zero_points: torch.Tensor | None = None
    data: dict | None = None


def finalize(
    model: HfModelHandler,
    output_model_path: str,
    wrapper: ModelWrapper,
    quant_config: OliveHfQuantizationConfig,
    device: str,
    retie_word_embeddings: bool = False,
) -> HfModelHandler:
    """Finalize quantization by replacing linear and embedding layers with their quantized counterparts.

    Args:
        model: The HuggingFace model to finalize.
        output_model_path: Path to save the finalized quantized model.
        wrapper: ModelWrapper containing the model to finalize.
        quant_config: Quantization configuration to use.
        device: Device to perform quantization on.
        retie_word_embeddings: Whether to retie word embeddings if they were originally tied and have compatible quantization.

    Returns:
        HfModelHandler with the finalized quantized model.

    """

    def should_quantize(module: torch.nn.Module, _: str) -> bool:
        return hasattr(module, "quant_info")

    def quantize_and_pack(module: torch.nn.Module, _: str) -> QuantLinear | QuantEmbedding:
        module.to(device)
        quant_cls = QuantEmbedding if isinstance(module, torch.nn.Embedding) else QuantLinear
        return quant_cls.from_module(
            module.to(device),
            bits=module.quant_info.quantizer.bits,
            symmetric=module.quant_info.quantizer.symmetric,
            group_size=module.quant_info.quantizer.group_size,
            scales=module.quant_info.scales,
            zero_points=module.quant_info.zero_points,
        ).to("cpu")  # move the original module to CPU

    replace_matching_submodules(
        wrapper.model,
        should_quantize,
        quantize_and_pack,
        description="Quantizing and packing linear layers",
    )

    if retie_word_embeddings:
        tie_quant_word_embeddings(wrapper.model)
        wrapper.config.tie_word_embeddings = True
        wrapper.model.config.tie_word_embeddings = True

    wrapper.model.quantization_method = quant_config.quant_method
    wrapper.model.config.quantization_config = quant_config

    # save the quantized model
    wrapper.model.save_pretrained(output_model_path)
    model.save_metadata(output_model_path)

    return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
