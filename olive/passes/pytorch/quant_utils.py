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
from olive.common.utils import get_attr, set_attr, tensor_data_to_device
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


def _root_module_name(name: str, name_prefix: str = "") -> str:
    """Return the module name relative to the saved model root."""
    return f"{name_prefix}{name}" if name else name_prefix.rstrip(".")


def _is_in_component(name: str, source_paths: list[str]) -> bool:
    """Return True if ``name`` (root-relative) belongs to any of the component's sub-trees."""
    if not source_paths:
        return True
    return any(name == path or name.startswith(f"{path}.") for path in source_paths)


def _get_component_source_paths(model: HfModelHandler) -> list[str]:
    """Return the component's dotted sub-module paths from model attributes.

    Reads ``component_source_paths`` (list, as written by ``ModelConfig.select_components``
    from the mobius plan). Falls back to the legacy singular ``component_source_path`` string
    for backward compatibility with models tagged by older Olive releases.
    """
    attributes = model.model_attributes or {}
    source_paths = attributes.get("component_source_paths")
    if source_paths:
        return list(source_paths)
    legacy = attributes.get("component_source_path")
    return [legacy] if legacy else []


def _component_slice_path(source_paths: list[str]) -> str | None:
    """Return the sub-module to slice and quantize for the given component paths.

    A single-path component slices to that path directly. A component spanning several
    disjoint sub-trees (e.g. ``model.layers`` + ``model.norm`` + ``lm_head``) slices to their
    greatest common dotted ancestor (the root when they share none), and ``_is_in_component``
    then restricts quantization to the declared sub-trees within that slice.
    """
    if not source_paths:
        return None
    if len(source_paths) == 1:
        return source_paths[0]
    split = [path.split(".") for path in source_paths]
    common: list[str] = []
    for segments in zip(*split):
        if len(set(segments)) == 1:
            common.append(segments[0])
        else:
            break
    return ".".join(common) or None


def _path_with_leaf(source_paths: list[str], leaves: set[str]) -> str | None:
    for path in source_paths:
        if path.rsplit(".", 1)[-1] in leaves:
            return path
    return None


def _module_path(root_model: torch.nn.Module, target: torch.nn.Module | None) -> str | None:
    if target is None:
        return None
    return next((name for name, module in root_model.named_modules() if module is target), None)


def _root_component_model_wrapper(
    root_model: torch.nn.Module,
    backbone_path: str,
    layers_path: str,
    source_paths: list[str],
    embedding_path: str | None = None,
) -> ModelWrapper:
    """Create a text wrapper whose structural paths are relative to the full HF model."""
    backbone = get_attr(root_model, backbone_path) if backbone_path else root_model
    wrapper = ModelWrapper(backbone.config)
    wrapper.LAYERS = {"default": layers_path}

    norm_path = _path_with_leaf(
        source_paths,
        {"norm", "final_layer_norm", "layer_norm"},
    )
    if norm_path is None:
        norm_path = _module_path(root_model, getattr(backbone, "norm", None))
    if norm_path is not None:
        wrapper.PRE_HEAD_LAYERNORM = {"default": norm_path}

    head_path = _path_with_leaf(
        source_paths,
        {"lm_head", "proj_out", "output_projection", "codec_head"},
    )
    if head_path is None:
        get_output_embeddings = getattr(root_model, "get_output_embeddings", None)
        output_embeddings = get_output_embeddings() if callable(get_output_embeddings) else None
        head_path = _module_path(root_model, output_embeddings)
    if head_path is not None:
        wrapper.LM_HEAD = {"default": head_path}

    if embedding_path is None:
        get_input_embeddings = getattr(backbone, "get_input_embeddings", None)
        input_embeddings = get_input_embeddings() if callable(get_input_embeddings) else None
        if input_embeddings is None:
            input_embeddings = getattr(backbone, "embed_tokens", None)
        embedding_path = _module_path(root_model, input_embeddings)
    if embedding_path is not None:
        wrapper.EMBEDDINGS = {"default": [embedding_path]}

    rotary_path = _module_path(root_model, getattr(backbone, "rotary_emb", None))
    if rotary_path is not None:
        wrapper.ROTARY_EMBEDDING = {"default": rotary_path}

    wrapper.set_model(root_model)
    return wrapper


class _GenericComponentWrapper(ModelWrapper):
    """Minimal wrapper for non-decoder components that do not expose transformer layers."""

    def __init__(self, model: torch.nn.Module, config) -> None:
        super().__init__(config)
        super().set_model(model, initialize_layer_wrappers=False)

    def maybe_untie_word_embeddings(self):
        if not getattr(self.config, "tie_word_embeddings", False):
            return

        root_model = self.olive_root_model if self.olive_root_model is not None else self.model

        # T5 reuses one Embedding module at several paths, while BART uses
        # distinct modules that share one Parameter. Split both forms before
        # replacing only the selected component.
        embedding_aliases: dict[int, list[tuple[str, torch.nn.Embedding]]] = {}
        for name, module in root_model.named_modules(remove_duplicate=False):
            if name and isinstance(module, torch.nn.Embedding):
                embedding_aliases.setdefault(id(module), []).append((name, module))
        for aliases in embedding_aliases.values():
            for name, module in aliases[1:]:
                set_attr(root_model, name, deepcopy(module))

        tied_weights: dict[int, list[torch.nn.Module]] = {}
        for module in root_model.modules():
            if isinstance(module, (torch.nn.Embedding, torch.nn.Linear)):
                tied_weights.setdefault(id(module.weight), []).append(module)
        for modules in tied_weights.values():
            if len(modules) < 2 or not any(isinstance(module, torch.nn.Embedding) for module in modules):
                continue
            for module in modules[1:]:
                weight = module.weight
                module.weight = torch.nn.Parameter(
                    weight.detach().clone(),
                    requires_grad=weight.requires_grad,
                )

        for module in root_model.modules():
            module_config = getattr(module, "config", None)
            if module_config is not None and hasattr(module_config, "tie_word_embeddings"):
                module_config.tie_word_embeddings = False
        self.config.tie_word_embeddings = False

    def get_embeds(self, return_name: bool = True):
        raise AttributeError("The selected component has no language-model embeddings.")

    def get_lm_head(self, return_name: bool = True):
        raise AttributeError("The selected component has no language-model head.")


def _component_model_wrapper(
    root_model: torch.nn.Module,
    source_paths: list[str],
    component_role: str | None,
) -> tuple[ModelWrapper, str]:
    """Wrap a selected component while retaining paths relative to the saved HF model.

    Decoder components can span disjoint runtime sub-trees (for example a nested
    language backbone plus a top-level ``lm_head``). In that case the wrapper must
    operate on the full model so replacement and saving preserve every component,
    while its structural lookups point at the selected decoder paths.
    """
    if not source_paths:
        if component_role not in {None, "decoder"}:
            return _GenericComponentWrapper(root_model, root_model.config), ""
        return ModelWrapper.from_model(root_model), ""

    if component_role not in {None, "decoder", "embedding"}:
        slice_path = _component_slice_path(source_paths)
        component_model = get_attr(root_model, slice_path) if slice_path else root_model
        config = getattr(component_model, "config", root_model.config)
        return _GenericComponentWrapper(component_model, config), (f"{slice_path}." if slice_path else "")

    slice_path = _component_slice_path(source_paths)
    embedding_path = _path_with_leaf(source_paths, {"embed_tokens", "shared"})
    if embedding_path is not None:
        backbone_path = embedding_path.rpartition(".")[0]
        backbone = get_attr(root_model, backbone_path) if backbone_path else root_model
        if hasattr(backbone, "layers"):
            layers_path = f"{backbone_path}.layers" if backbone_path else "layers"
            wrapper = _root_component_model_wrapper(
                root_model,
                backbone_path,
                layers_path,
                source_paths,
                embedding_path,
            )
            return wrapper, ""

    layers_path = _path_with_leaf(source_paths, {"layers"})
    if layers_path is not None and (component_role is not None or not slice_path):
        backbone_path = layers_path.rpartition(".")[0]
        return _root_component_model_wrapper(
            root_model,
            backbone_path,
            layers_path,
            source_paths,
        ), ""

    if component_role in {"decoder", "embedding"}:
        component_model = get_attr(root_model, slice_path) if slice_path else root_model
        config = getattr(component_model, "config", root_model.config)
        return _GenericComponentWrapper(component_model, config), (f"{slice_path}." if slice_path else "")

    if slice_path:
        quant_model = get_attr(root_model, slice_path)
        return ModelWrapper.from_model(quant_model), f"{slice_path}."

    return ModelWrapper.from_model(root_model), ""


def _validate_component_source_paths(
    root_model: torch.nn.Module,
    source_paths: list[str],
) -> None:
    missing = [path for path in source_paths if get_attr(root_model, path) is None]
    if missing:
        raise ValueError(
            "Component source path(s) do not exist in the loaded Hugging Face model: "
            f"{missing}. The paths must match runtime model.named_modules() names."
        )


def get_qkv_quantization_groups(
    wrapper: ModelWrapper,
    module_names: set[str] | None = None,
    name_prefix: str = "",
) -> list[tuple[str, ...]]:
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
        attn_input_names = (module_to_name.get(id(module)) for module in attn_inputs)
        group = tuple(
            root_name
            for root_name in (_root_module_name(name, name_prefix) for name in attn_input_names)
            if root_name and (module_names is None or root_name in module_names)
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
    module_names: set[str] | None = None,
    name_prefix: str = "",
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
    for group in get_qkv_quantization_groups(wrapper, module_names=module_names, name_prefix=name_prefix):
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

    component_source_paths = _get_component_source_paths(model)
    component_attributes = model.model_attributes or {}
    component_name = component_attributes.get("component_name")
    component_role = component_attributes.get("component_role")
    if component_name and component_name != "model" and not component_source_paths:
        raise ValueError(
            f"Component {component_name!r} has no runtime source paths; refusing to apply "
            "a PyTorch quantization pass to the whole model."
        )
    root_model = load_hf_base_model(model)
    _validate_component_source_paths(root_model, component_source_paths)
    wrapper, name_prefix = _component_model_wrapper(
        root_model,
        component_source_paths,
        component_role,
    )
    wrapper.olive_root_model = root_model
    wrapper.olive_component_path = name_prefix.rstrip(".") or None
    wrapper.olive_component_role = component_role
    wrapper.model.eval()

    excluded_attn_inputs = _collect_excluded_attn_inputs(wrapper) if exclude_attn_inputs else set()

    selected_module_names = {_root_module_name(name, name_prefix) for name, _ in wrapper.model.named_modules()}
    fresh_qcfg = normalize_qkv_quant_config(
        wrapper,
        get_quant_config(model, config),
        module_names=selected_module_names,
        name_prefix=name_prefix,
    )

    originally_tied_embeddings = getattr(wrapper.config, "tie_word_embeddings", False)
    if fresh_qcfg.lm_head or fresh_qcfg.embeds:
        wrapper.maybe_untie_word_embeddings()

    declared_head_name = _path_with_leaf(
        component_source_paths,
        {"lm_head", "proj_out", "output_projection", "codec_head", "output"},
    )
    try:
        lm_head_name = _root_module_name(wrapper.get_lm_head()[1], name_prefix)
    except AttributeError:
        lm_head_name = declared_head_name
        if fresh_qcfg.lm_head and lm_head_name is None:
            raise
    declared_embeds_name = _path_with_leaf(
        component_source_paths,
        {"embed_tokens", "shared", "tok_embeddings", "text_embedding", "codec_embedding"},
    )
    component_embedding_names = [
        name
        for name, module in root_model.named_modules()
        if isinstance(module, torch.nn.Embedding) and _is_in_component(name, component_source_paths)
    ]
    try:
        embeds_name = _root_module_name(wrapper.get_embeds()[1][0], name_prefix)
    except AttributeError:
        embeds_name = declared_embeds_name or next(
            (
                name
                for name in component_embedding_names
                if name.rsplit(".", 1)[-1]
                in {"embed_tokens", "shared", "tok_embeddings", "text_embedding", "codec_embedding"}
            ),
            component_embedding_names[0] if len(component_embedding_names) == 1 else None,
        )
        if fresh_qcfg.embeds and not component_embedding_names:
            raise ValueError("The selected component has no torch.nn.Embedding modules to quantize.") from None

    def should_quantize(module: torch.nn.Module, name: str) -> bool:
        root_name = _root_module_name(name, name_prefix)
        if module in excluded_attn_inputs:
            return False
        # When the slice spans more than the component (multi-path components slice to a
        # common ancestor), restrict quantization to modules inside the declared sub-trees.
        if not _is_in_component(root_name, component_source_paths):
            return False
        if isinstance(module, torch.nn.Linear):
            return lm_head_name is None or root_name != lm_head_name or fresh_qcfg.lm_head
        if fresh_qcfg.embeds and isinstance(module, torch.nn.Embedding):
            if component_source_paths or isinstance(wrapper, _GenericComponentWrapper):
                return root_name in component_embedding_names
            return root_name == embeds_name
        return False

    fresh_names = {
        _root_module_name(name, name_prefix)
        for name, module in wrapper.model.named_modules()
        if should_quantize(module, name)
    }

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
            _root_module_name(name, name_prefix)
            for name, module in wrapper.model.named_modules()
            if isinstance(module, (QuantLinear, QuantEmbedding))
        }
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
        qcfg = normalize_qkv_quant_config(
            wrapper,
            qcfg,
            locked_modules=already_quantized,
            module_names=selected_module_names,
            name_prefix=name_prefix,
        )
    else:
        qcfg = fresh_qcfg

    existing_modules_to_not_convert = {
        excluded
        for excluded in qcfg.modules_to_not_convert or []
        if not any(excluded in fresh_name for fresh_name in fresh_names)
    }
    unquantized_modules = {
        name
        for name, module in root_model.named_modules()
        if isinstance(module, (torch.nn.Linear, torch.nn.Embedding)) and name not in fresh_names
    }
    qcfg.modules_to_not_convert = sorted(existing_modules_to_not_convert | unquantized_modules) or None

    new_qargs: dict[str, dict[str, int | bool]] = {}

    def add_quant_info(module: torch.nn.Module, name: str) -> torch.nn.Module:
        # TODO(jambayk): validate that the module and config are compatible
        root_name = _root_module_name(name, name_prefix)
        qargs = qcfg.get_qlinear_init_args(root_name)
        module.quant_info = QuantInfo(quantizer=WeightQuantizer(**qargs))
        new_qargs[root_name] = qargs
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
        and not isinstance(wrapper, _GenericComponentWrapper)
        and lm_head_name is not None
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
    component_role = wrapper.olive_component_role
    if component_role not in {None, "decoder"} or isinstance(wrapper, _GenericComponentWrapper):
        raise ValueError(
            "Layerwise calibration requires a decoder component with identifiable transformer layers. "
            "Use RTN or KQuant for generic encoder/vision/embedding components."
        )

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

    root_model = wrapper.olive_root_model if wrapper.olive_root_model is not None else wrapper.model
    packed_model = replace_matching_submodules(
        wrapper.model,
        should_quantize,
        quantize_and_pack,
        description="Quantizing and packing linear layers",
    )
    if packed_model is not wrapper.model:
        component_path = wrapper.olive_component_path
        if component_path:
            set_attr(root_model, component_path, packed_model)
        else:
            root_model = packed_model

    if retie_word_embeddings:
        tie_quant_word_embeddings(packed_model)
        quant_config.tie_word_embeddings = True

    root_model.quantization_method = quant_config.quant_method
    root_model.config.quantization_config = quant_config

    # save the quantized model
    root_model.save_pretrained(output_model_path)
    model.save_metadata(output_model_path)

    return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
