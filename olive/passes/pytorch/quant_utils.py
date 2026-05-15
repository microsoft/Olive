# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import torch

from olive.common.hf.wrapper import ModelWrapper
from olive.common.quant.hf_utils import (
    OliveHfQuantizationConfig,
    OliveHfQuantizationMethod,
    replace_matching_submodules,
    tie_quant_word_embeddings,
)
from olive.common.quant.nn import QuantEmbedding, QuantLinear
from olive.common.quant.patterns import match_skip
from olive.common.quant.tensor import QuantTensor
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


def get_quantizer_config(allow_embeds: bool = False, allow_moe: bool = False) -> dict[str, PassConfigParam]:
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
        **(
            {
                "moe": PassConfigParam(
                    type_=bool,
                    default_value=False,
                    description=(
                        "Whether to quantize MoE expert modules / parameters. When False (default), every "
                        "nn.Module under each experts subtree is skipped. Defaults reuse the pass-level "
                        "bits/group_size/sym settings; use ``overrides`` to tune experts independently."
                    ),
                )
            }
            if allow_moe
            else {}
        ),
        "modules_to_not_convert": PassConfigParam(
            type_=list,
            default_value=None,
            description=(
                "Optional list of module name patterns to exclude from quantization. Plain strings use"
                " substring matching (HF semantics); entries prefixed with 're:' use re.fullmatch."
            ),
        ),
        "overrides": PassConfigParam(
            type_=dict,
            default_value=None,
            description=(
                "Optional dictionary to specify overrides for specific modules. The keys are module names"
                " (literal match) or 're:<regex>' patterns (regex fullmatch). Values are dictionaries with"
                " any of the following keys: 'bits', 'symmetric', 'group_size'. These overrides take"
                " precedence over the overrides provided in the mixed precision info."
            ),
        ),
    }


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
        if not isinstance(existing_qcfg, dict):
            existing_qcfg = existing_qcfg.to_dict()
        if existing_qcfg.get("quant_method", None) != OliveHfQuantizationMethod.OLIVE:
            raise ValueError("Model has an existing quantization configuration that is not compatible with this pass.")

    wrapper = ModelWrapper.from_model(load_hf_base_model(model))
    wrapper.model.eval()

    qcfg = get_quant_config(model, config)

    originally_tied_embeddings = wrapper.config.tie_word_embeddings
    if qcfg.lm_head or qcfg.embeds:
        wrapper.maybe_untie_word_embeddings()

    lm_head_name = wrapper.get_lm_head()[1]
    embeds_name = wrapper.get_embeds()[1][0]
    new_qargs: dict[str, dict[str, int | bool]] = {}

    excluded_attn_inputs: set[torch.nn.Module] = set()
    if exclude_attn_inputs:
        for layer_wrapper in wrapper.get_layer_wrappers():
            attn_inputs, _ = layer_wrapper.get_attention_inputs()
            if len(attn_inputs) == 1:
                excluded_attn_inputs.add(attn_inputs[0])
            else:
                excluded_attn_inputs.update(attn_inputs[:2])

    # Collect every ``nn.Module`` under any experts subtree, so we can
    # honour the ``moe`` category flag the same way ``lm_head`` /
    # ``embeds`` are honoured today.
    expert_module_ids: set[int] = set()
    expert_owners: list[tuple[torch.nn.Module, str]] = []  # (experts_module, dotted_name)
    for layer_wrapper in wrapper.get_layer_wrappers():
        experts_module = layer_wrapper.get_experts(return_name=False)
        if experts_module is None:
            continue
        # Find the dotted name of this experts module relative to the model root.
        experts_name = None
        for name, mod in wrapper.model.named_modules():
            if mod is experts_module:
                experts_name = name
                break
        expert_owners.append((experts_module, experts_name or ""))
        for sub in experts_module.modules():
            expert_module_ids.add(id(sub))

    skip_patterns = list(getattr(qcfg, "modules_to_not_convert", None) or [])

    def should_quantize(module: torch.nn.Module, name: str) -> bool:
        if module in excluded_attn_inputs:
            return False
        if match_skip(name, skip_patterns):
            return False
        # category-flag skips (lm_head / embeds / moe) — first rule wins
        if id(module) in expert_module_ids and not getattr(qcfg, "moe", False):
            return False
        if isinstance(module, torch.nn.Linear):
            return name != lm_head_name or qcfg.lm_head
        if qcfg.embeds and isinstance(module, torch.nn.Embedding):
            return name == embeds_name
        return False

    def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
        # TODO(jambayk): validate that the module and config are compatible
        qargs = qcfg.get_qlinear_init_args(name)
        module.quant_info = QuantInfo(quantizer=WeightQuantizer(**qargs))
        new_qargs[name] = qargs
        return module

    replace_matching_submodules(wrapper.model, should_quantize, add_quant_info, description="Preparing model")

    # Fused-3D MoE: experts modules expose 3D nn.Parameters directly (e.g.
    # ``gate_up_proj`` of shape ``(num_experts, *, *)``). Annotate the
    # experts module with a per-parameter quant_info_3d dict so
    # ``finalize`` can replace each parameter with a QuantTensor.
    if getattr(qcfg, "moe", False):
        for experts_module, experts_name in expert_owners:
            param_qinfos: dict[str, QuantInfo] = {}
            for pname, param in experts_module.named_parameters(recurse=False):
                if param.dim() != 3:
                    continue
                full_name = f"{experts_name}.{pname}" if experts_name else pname
                if match_skip(full_name, skip_patterns):
                    continue
                qargs = qcfg.get_qlinear_init_args(full_name)
                param_qinfos[pname] = QuantInfo(quantizer=WeightQuantizer(**qargs))
                new_qargs[full_name] = qargs
            if param_qinfos:
                experts_module.quant_info_3d = param_qinfos

    # remove overrides for modules not being quantized
    for name in list(qcfg.overrides or {}):
        # ``re:`` keys aren't tied to a specific module, so leave them in place.
        if name.startswith("re:"):
            continue
        if name not in new_qargs:
            qcfg.overrides.pop(name)

    # merge the new_quant_settings into the existing quant_config
    if existing_qcfg:
        merged_qcfg_dict = existing_qcfg
        merged_qcfg_dict["overrides"] = existing_qcfg.get("overrides") or {}
        for name, qargs in new_qargs.items():
            override = {k: v for k, v in qargs.items() if merged_qcfg_dict[k] != v}
            if override:
                merged_qcfg_dict["overrides"][name] = override
        merged_qcfg_dict["lm_head"] |= qcfg.lm_head
        merged_qcfg_dict["embeds"] |= qcfg.embeds
        merged_qcfg_dict["moe"] = merged_qcfg_dict.get("moe", False) or getattr(qcfg, "moe", False)
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
        "moe": getattr(config, "moe", False),
        "modules_to_not_convert": getattr(config, "modules_to_not_convert", None) or [],
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

    # Fused-3D MoE: experts modules carry a per-parameter ``quant_info_3d``
    # dict. Replace each parameter with a 3D ``QuantTensor`` parameter.
    # The host module's forward continues to find the parameter at the
    # same attribute name; eager forwards may go through QuantTensor's
    # ``__getitem__`` / matmul fallbacks. ONNX export of fused-3D MoE is
    # deferred to Mobius — Olive only persists the checkpoint.
    for sub in wrapper.model.modules():
        info_3d: dict = getattr(sub, "quant_info_3d", None)
        if not info_3d:
            continue
        for pname, qinfo in info_3d.items():
            param = getattr(sub, pname)
            if not isinstance(param, torch.nn.Parameter):
                continue
            with torch.no_grad():
                quantizer = qinfo.quantizer
                qt = QuantTensor.from_float(
                    param.detach().to(device).to(param.dtype),
                    bits=quantizer.bits,
                    symmetric=quantizer.symmetric,
                    group_size=quantizer.group_size,
                ).to("cpu")
            sub._parameters[pname] = torch.nn.Parameter(qt, requires_grad=False)  # pylint: disable=W0212
        # Tidy up the marker so downstream save/load doesn't trip over it.
        delattr(sub, "quant_info_3d")

    if retie_word_embeddings:
        tie_quant_word_embeddings(wrapper.model)
        quant_config.tie_word_embeddings = True

    wrapper.model.quantization_method = quant_config.quant_method
    wrapper.model.config.quantization_config = quant_config

    # Flatten any QuantTensor parameters into plain ``register_buffer`` entries so
    # ``save_pretrained`` (which writes safetensors) can serialize them.
    # Naming convention: ``<param_name>_qweight`` / ``_scales`` / ``_qzeros``
    # sit on the same parent module as the original parameter. This matches
    # the suffix-style layout used by other prequantized formats and lets
    # downstream loaders (Mobius) discover the buffers without a registry.
    flatten_quant_tensor_params(wrapper.model)

    # save the quantized model
    wrapper.model.save_pretrained(output_model_path)
    model.save_metadata(output_model_path)

    return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)


def flatten_quant_tensor_params(module: torch.nn.Module) -> None:
    """Replace every ``QuantTensor`` ``nn.Parameter`` with plain registered buffers.

    For each parameter ``<name>`` whose data is a :class:`QuantTensor`, the
    parameter is deleted and the inner tensors are re-attached as
    ``<name>_qweight``, ``<name>_scales`` (and ``<name>_qzeros`` when
    asymmetric) on the same parent module. Additional metadata
    (``bits``, ``group_size``, ``symmetric``, ``shape``) is stored on
    ``<name>_quant_meta`` for round-trip loading.

    This is used at save time so safetensors / pickle never sees a
    tensor subclass.
    """
    for sub in module.modules():
        names = [n for n, p in sub.named_parameters(recurse=False) if isinstance(p.data, QuantTensor)]
        for name in names:
            qt: QuantTensor = sub._parameters[name].data  # type: ignore[assignment]
            del sub._parameters[name]
            sub.register_buffer(f"{name}_qweight", qt.qweight.detach().clone())
            sub.register_buffer(f"{name}_scales", qt.scales.detach().clone())
            if qt.qzeros is not None:
                sub.register_buffer(f"{name}_qzeros", qt.qzeros.detach().clone())
            # ``register_buffer`` requires Tensors, so the metadata lives
            # outside the state_dict as a plain attribute. It is consumed
            # by ``OliveHfQuantizer`` at load time via the
            # ``quantization_config`` for shape/bits and is also encoded
            # in the parameter's full name + scales shape if needed.
            setattr(
                sub,
                f"{name}_quant_meta",
                {
                    "bits": qt.bits,
                    "group_size": qt.group_size,
                    "symmetric": qt.symmetric,
                    "shape": tuple(qt.shape),
                },
            )
