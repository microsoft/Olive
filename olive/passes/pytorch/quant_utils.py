# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import (
    OliveHfQuantizationConfig,
    OliveHfQuantizationMethod,
    OliveHfQuantizationOverrideConfig,
    replace_matching_submodules,
    tie_quant_word_embeddings,
)
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import tensor_data_to_device
from olive.constants import PrecisionBits
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import get_calibration_dataset, load_hf_base_model

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
                " values are dictionaries with any of the following keys: 'bits', 'symmetric', 'group_size'. These"
                " overrides take precedence over the overrides provided in the mixed precision info."
            ),
        ),
    }


def get_qkv_quantization_groups(wrapper: ModelWrapper, module_names: set[str] | None = None) -> list[tuple[str, ...]]:
    """Get attention input projection groups that must share quantization settings.

    Names are resolved from ``wrapper.model.named_modules()`` to stay correct for any layer
    container (``ModuleList``, ``ModuleDict``, custom containers) and for unpacked QKV
    submodules. When ``module_names`` is provided, attention inputs not in the set are
    dropped from the group. Groups with fewer than two members are skipped.
    """
    module_to_name = {id(module): name for name, module in wrapper.model.named_modules()}
    qkv_groups = []
    for layer_wrapper in wrapper.get_layer_wrappers():
        attn_inputs, _ = layer_wrapper.get_attention_inputs()
        group = tuple(
            name
            for name in (module_to_name.get(id(module)) for module in attn_inputs)
            if name is not None and (module_names is None or name in module_names)
        )
        if len(group) > 1:
            qkv_groups.append(group)
    return qkv_groups


def _quant_config_rank(qargs: dict[str, int | bool]) -> tuple[int, int, int]:
    """Rank quantization configs by precision; higher rank means more precise.

    Ordering: higher ``bits`` wins; among equal bits, smaller positive ``group_size`` wins;
    per-channel (``-1``) wins over per-tensor (``0``) but loses to positive group sizes.
    ``symmetric`` is intentionally not part of the ordering since it is a representation
    choice rather than a strict precision axis.
    """
    bits = qargs["bits"].value if hasattr(qargs["bits"], "value") else qargs["bits"]
    group_size = qargs["group_size"]
    if group_size > 0:
        group_size_rank = (2, -group_size)
    elif group_size == -1:
        group_size_rank = (1, 0)
    else:
        group_size_rank = (0, 0)
    return bits, *group_size_rank


def normalize_qkv_quant_config(
    wrapper: ModelWrapper,
    qcfg: OliveHfQuantizationConfig,
    locked_modules: set[str] | None = None,
) -> OliveHfQuantizationConfig:
    """Promote split QKV projection overrides to one shared quantization config.

    Groups span all attention input projections of a layer regardless of whether the current
    pass quantizes them; follow-up passes (e.g. RTN after AutoClip) will pick up the shared
    settings via the recorded overrides so downstream QKV fusion remains valid.

    ``locked_modules`` are modules whose overrides must not be rewritten -- typically the
    pre-existing overrides of an already-quantized checkpoint. For a group containing a
    locked member, the shared config is forced to that locked member's config; if multiple
    locked members of one group disagree, the group is left untouched.
    """
    locked_modules = locked_modules or set()
    for group in get_qkv_quantization_groups(wrapper):
        group_qargs = {name: qcfg.get_qlinear_init_args(name) for name in group}
        if len({tuple(qargs.items()) for qargs in group_qargs.values()}) == 1:
            continue

        locked_in_group = [name for name in group if name in locked_modules]
        locked_configs = {tuple(group_qargs[name].items()) for name in locked_in_group}
        if len(locked_configs) > 1:
            logger.debug(
                "QKV group %s contains already-quantized members with conflicting configs; "
                "skipping (downstream QKV fusion may be inhibited).",
                group,
            )
            continue
        promoted_qargs = (
            group_qargs[locked_in_group[0]] if locked_in_group else max(group_qargs.values(), key=_quant_config_rank)
        )

        logger.debug("Promoting QKV group %s to shared quantization config %s", group, promoted_qargs)
        for name in group:
            if name in locked_modules:
                continue
            override = {k: v for k, v in promoted_qargs.items() if getattr(qcfg, k) != v}
            if override:
                qcfg.overrides[name] = OliveHfQuantizationOverrideConfig(**override)
            else:
                qcfg.overrides.pop(name, None)

    return qcfg


def _collect_excluded_attn_inputs(wrapper: ModelWrapper) -> set[torch.nn.Module]:
    excluded: set[torch.nn.Module] = set()
    for layer_wrapper in wrapper.get_layer_wrappers():
        attn_inputs, _ = layer_wrapper.get_attention_inputs()
        if len(attn_inputs) == 1:
            excluded.add(attn_inputs[0])
        else:
            excluded.update(attn_inputs[:2])
    return excluded


def prepare_model(
    model: HfModelHandler,
    config: type[BasePassConfig],
    allow_quantized: bool = False,
    exclude_attn_inputs: bool = False,
) -> tuple[ModelWrapper, OliveHfQuantizationConfig, bool]:
    """Prepare the model for quantization by adding quant_info to linear layers.

    Args:
        model: The HuggingFace model to prepare.
        config: Configuration object containing quantization parameters.
        allow_quantized: Whether to allow already (partially) quantized models.
        exclude_attn_inputs: Whether to exclude attention input projection layers from quantization.

    Returns:
        A tuple containing ModelWrapper with prepared model, the quantization configuration, and a boolean indicating if the word embeddings are eligible for tieing.

    """
    if existing_qcfg := getattr(model.get_hf_model_config(), "quantization_config", None):
        if not allow_quantized:
            raise ValueError("Model is already quantized. Cannot quantize again using this pass.")
        # Always work on a fresh copy: the underlying HF config holds the original object
        # (dict or dataclass) and we mutate ``existing_qcfg`` heavily below.
        existing_qcfg = deepcopy(existing_qcfg) if isinstance(existing_qcfg, dict) else existing_qcfg.to_dict()
        if existing_qcfg.get("quant_method", None) != OliveHfQuantizationMethod.OLIVE:
            raise ValueError("Model has an existing quantization configuration that is not compatible with this pass.")

    wrapper = ModelWrapper.from_model(load_hf_base_model(model))
    wrapper.model.eval()

    excluded_attn_inputs = _collect_excluded_attn_inputs(wrapper) if exclude_attn_inputs else set()

    fresh_qcfg = normalize_qkv_quant_config(wrapper, get_quant_config(model, config))

    originally_tied_embeddings = wrapper.config.tie_word_embeddings
    if fresh_qcfg.lm_head or fresh_qcfg.embeds:
        wrapper.maybe_untie_word_embeddings()

    lm_head_name = wrapper.get_lm_head()[1]
    embeds_name = wrapper.get_embeds()[1][0]

    def should_quantize(module: torch.nn.Module, name: str) -> bool:
        if module in excluded_attn_inputs:
            return False
        if isinstance(module, torch.nn.Linear):
            return name != lm_head_name or fresh_qcfg.lm_head
        if fresh_qcfg.embeds and isinstance(module, torch.nn.Embedding):
            return name == embeds_name
        return False

    # Pre-existing quantized weights are immutable. If we're merging with an existing
    # checkpoint, build the final qcfg first (merge fresh into existing, then renormalize
    # QKV with already-quantized modules locked) so that the quant_info we attach below
    # uses the same settings the on-disk fusion will require. Every module that is already
    # a QuantLinear/QuantEmbedding after load is on-disk-immutable, including those that
    # used the existing config's defaults (no explicit override entry).
    on_disk_overrides: set[str] = set()
    already_quantized: set[str] = set()
    if existing_qcfg:
        on_disk_overrides = set((existing_qcfg.get("overrides") or {}).keys())
        already_quantized = {
            name for name, module in wrapper.model.named_modules() if isinstance(module, (QuantLinear, QuantEmbedding))
        }
        fresh_names = {name for name, module in wrapper.model.named_modules() if should_quantize(module, name)}
        merged = existing_qcfg
        merged["overrides"] = existing_qcfg.get("overrides") or {}
        for name in fresh_names:
            qargs = fresh_qcfg.get_qlinear_init_args(name)
            override = {k: v for k, v in qargs.items() if merged[k] != v}
            if override:
                merged["overrides"][name] = override
        merged["lm_head"] |= fresh_qcfg.lm_head
        merged["embeds"] |= fresh_qcfg.embeds
        qcfg = OliveHfQuantizationConfig(**merged)
        qcfg = normalize_qkv_quant_config(wrapper, qcfg, locked_modules=already_quantized)
    else:
        qcfg = fresh_qcfg

    new_qargs: dict[str, dict[str, int | bool]] = {}

    def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
        # TODO(jambayk): validate that the module and config are compatible
        qargs = qcfg.get_qlinear_init_args(name)
        module.quant_info = QuantInfo(quantizer=WeightQuantizer(**qargs))
        new_qargs[name] = qargs
        return module

    replace_matching_submodules(wrapper.model, should_quantize, add_quant_info, description="Preparing model")

    # Drop overrides for modules that won't be quantized this pass. Pre-existing (on-disk)
    # overrides are preserved verbatim since they describe already-quantized weights.
    # QKV-group overrides for modules excluded from this pass are not kept: when the
    # follow-up pass runs, the quantized members in the group will be locked and pull the
    # remaining members back into the shared config via ``normalize_qkv_quant_config``.
    for name in list(qcfg.overrides or {}):
        if name not in new_qargs and name not in on_disk_overrides:
            qcfg.overrides.pop(name)

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
        # merge overrides, user provided overrides take precedence
        for name, override in mp_info.get("overrides", {}).items():
            merged = override.copy()
            merged.update({k: v for k, v in quant_config["overrides"].get(name, {}).items() if v is not None})
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


@torch.no_grad()
def get_layer_inputs_for_calibration(
    model: HfModelHandler,
    wrapper: ModelWrapper,
    data_config,
    device: str,
) -> tuple[list[torch.Tensor], list[tuple], list[dict]]:
    """Get initial layer inputs for calibration.

    Args:
        model: The HuggingFace model handler.
        wrapper: ModelWrapper containing the model.
        data_config: Data config used to build the calibration dataset.
        device: Device to run calibration on.

    Returns:
        Tuple containing hidden states, layer args, and layer kwargs.

    """
    hidden_states, layer_args, layer_kwargs = [], [], []

    pre_layer_modules = list(wrapper.get_embeds(return_name=False))
    if rotary_embed := wrapper.get_rotary_embed(return_name=False):
        pre_layer_modules.append(rotary_embed)
    for module in pre_layer_modules:
        module.to(device)

    def store_input_hook(_, args: tuple, kwargs: dict) -> None:
        if kwargs.get("hidden_states") is not None:
            args = (kwargs.pop("hidden_states"), *args)
        hidden_states.append(args[0])
        layer_args.append(args[1:])
        layer_kwargs.append(kwargs)
        raise ValueError

    first_layer = wrapper.get_layers(return_name=False)[0]
    hook = first_layer.register_forward_pre_hook(store_input_hook, with_kwargs=True)

    for data in get_calibration_dataset(model, data_config):
        try:
            wrapper.model(**tensor_data_to_device(data, device))
        except ValueError:
            pass

    hook.remove()
    for module in pre_layer_modules:
        module.to("cpu")

    return hidden_states, layer_args, layer_kwargs


@torch.no_grad()
def run_layer(
    layer: torch.nn.Module,
    hidden_states: list[torch.Tensor],
    layer_args: list[tuple] | None = None,
    layer_kwargs: list[dict] | None = None,
    return_output: bool = False,
) -> list[torch.Tensor] | None:
    """Run a layer with the given inputs.

    Args:
        layer: The model layer to run.
        hidden_states: List of hidden state tensors.
        layer_args: List of additional positional arguments for each input.
        layer_kwargs: List of keyword arguments for each input.
        return_output: Whether to return the layer outputs.

    Returns:
        List of output tensors if return_output is True, otherwise None.

    """
    outputs = []
    layer.to(hidden_states[0].device)

    for i, hs in enumerate(hidden_states):
        layer_output = layer(
            hs,
            *(layer_args[i] if layer_args else ()),
            **(layer_kwargs[i] if layer_kwargs else {}),
        )
        if return_output:
            if isinstance(layer_output, tuple):
                layer_output = layer_output[0]
            outputs.append(layer_output)

    layer.to("cpu")
    return outputs or None


@torch.no_grad()
def run_layerwise_quantization(
    model: HfModelHandler,
    wrapper: ModelWrapper,
    data_config,
    input_hook: Callable[[torch.nn.Module, tuple, torch.Tensor], None],
    process_module: Callable[[torch.nn.Module, str], None],
    update_before_process: bool,
    include_lm_head: bool,
    device: str | None = None,
) -> str:
    """Run a layerwise calibration + processing loop with configurable hook order.

    Args:
        model: The HuggingFace model handler.
        wrapper: ModelWrapper containing the model.
        data_config: Data config used to build the calibration dataset.
        input_hook: Forward hook to collect calibration data for a module.
        process_module: Callback to process a single module after hooks are collected.
        update_before_process: Whether to run the layer forward to get next inputs before processing.
        include_lm_head: Whether to process the lm_head similarly to other layers.
        device: Device to run calibration on. If None, uses cuda when available.

    Returns:
        Device string used for calibration.

    """
    from tqdm.auto import tqdm

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    original_use_cache = getattr(wrapper.model.config, "use_cache", None)
    if original_use_cache is not None:
        wrapper.model.config.use_cache = False

    hidden_states, layer_args, layer_kwargs = get_layer_inputs_for_calibration(model, wrapper, data_config, device)
    if not hidden_states:
        raise ValueError("Calibration data is empty. Provide a valid data_config.")

    total_steps = wrapper.num_hidden_layers + (1 if include_lm_head else 0)
    pbar = tqdm(total=total_steps, desc="Processing layers...")

    for layer_idx, layer in enumerate(wrapper.get_layers(return_name=False)):
        pbar.set_postfix(module=f"layers.{layer_idx}", refresh=False)
        quantizable_modules = [module for module in layer.modules() if hasattr(module, "quant_info")]
        handles = [module.register_forward_hook(input_hook) for module in quantizable_modules]

        if update_before_process:
            hidden_states = run_layer(
                layer,
                hidden_states,
                layer_args,
                layer_kwargs,
                return_output=True,
            )
        else:
            run_layer(layer, hidden_states, layer_args, layer_kwargs)

        for handle in handles:
            handle.remove()

        for module in quantizable_modules:
            process_module(module, device)

        if not update_before_process:
            hidden_states = run_layer(
                layer,
                hidden_states,
                layer_args,
                layer_kwargs,
                return_output=True,
            )

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        pbar.update(1)

    if include_lm_head:
        hidden_states = run_layer(wrapper.get_pre_head_layernorm(return_name=False), hidden_states, return_output=True)
        lm_head = wrapper.get_lm_head(return_name=False)
        pbar.set_postfix(module="lm_head", refresh=False)
        handle = lm_head.register_forward_hook(input_hook)
        run_layer(lm_head, hidden_states, return_output=True)
        handle.remove()
        process_module(lm_head, device)
        pbar.update(1)

    pbar.close()

    if original_use_cache is not None:
        wrapper.model.config.use_cache = original_use_cache

    return device


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
        quant_config.tie_word_embeddings = True

    wrapper.model.quantization_method = quant_config.quant_method
    wrapper.model.config.quantization_config = quant_config

    # save the quantized model
    wrapper.model.save_pretrained(output_model_path)
    model.save_metadata(output_model_path)

    return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
