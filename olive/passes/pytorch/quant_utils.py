# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
from __future__ import annotations

import contextlib
import inspect
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
    tie_quant_word_embeddings,
)
from olive.common.quant.selection import iter_quant_targets
from olive.common.quant.state_dict import install_quant_tensor_param
from olive.common.quant.tensor import QuantTensor
from olive.common.quant.utils import WeightQuantizer
from olive.common.utils import get_attr, set_attr, tensor_data_to_device
from olive.constants import PrecisionBits
from olive.passes.pass_config import PassConfigParam
from olive.passes.pytorch.common import inherit_hf_from_hf
from olive.passes.pytorch.train_utils import get_calibration_dataset, load_hf_base_model
from olive.search.search_parameter import Boolean, Categorical

if TYPE_CHECKING:
    from olive.model import HfModelHandler
    from olive.passes.pass_config import BasePassConfig
    from olive.passes.pytorch.moe_calib import MoeCalibrationSession


logger = logging.getLogger(__name__)


def get_quantizer_config(allow_embeds: bool = False, allow_moe: bool = False) -> dict[str, PassConfigParam]:
    return {
        "bits": PassConfigParam(
            type_=PrecisionBits,
            default_value=PrecisionBits.BITS4,
            search_defaults=Categorical([PrecisionBits.BITS2, PrecisionBits.BITS4, PrecisionBits.BITS8]),
            description="quantization bits. Default value is 4",
        ),
        "group_size": PassConfigParam(
            type_=int,
            default_value=128,
            search_defaults=Categorical([-1, 16, 32, 64, 128]),
            description="Block size for quantization. Default value is 128.",
        ),
        "sym": PassConfigParam(
            type_=bool,
            default_value=False,
            search_defaults=Boolean(),
            description="Symmetric quantization. Default value is False.",
        ),
        "lm_head": PassConfigParam(
            type_=bool,
            default_value=False,
            search_defaults=Boolean(),
            description="Whether to quantize the language model head. Default value is False.",
        ),
        "quantize_vision": PassConfigParam(
            type_=bool,
            default_value=False,
            description=(
                "Whether to quantize a composite vision-language model's vision tower in this pass. When "
                "False (default), the vision tower (``visual``/``vision_tower``/``vision_model``/"
                "``vision_encoder``) is left in full precision -- the typical Olive pipeline quantizes it "
                "separately downstream (e.g. on the ONNX side). Set to True to quantize the vision tower "
                "here too, e.g. when this pass is the only quantization step for the model."
            ),
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

    Reads ``component_source_paths`` from model attributes. Falls back to the legacy singular
    ``component_source_path`` string for backward compatibility.
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
        # partial_ok: MLA-style attentions (deepseek_v3) expose only a subset of
        # q/k/v_proj; QKV grouping treats the result as an unordered set, so dropping the
        # missing entries is safe here (unlike the positional consumers in rotate.py).
        attn_inputs, _ = layer_wrapper.get_attention_inputs(partial_ok=True)
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
        # partial_ok: MLA-style attentions (deepseek_v3) expose only a subset of
        # q/k/v_proj; QKV grouping treats the result as an unordered set, so dropping the
        # missing entries is safe here (unlike the positional consumers in rotate.py).
        attn_inputs, _ = layer_wrapper.get_attention_inputs(partial_ok=True)
        if len(attn_inputs) == 1:
            excluded.add(attn_inputs[0])
        else:
            excluded.update(attn_inputs[:2])
    return excluded


def _collect_already_quantized_names(model: torch.nn.Module) -> set[str]:
    """Return override-key names of parameters already backed by a ``QuantTensor``.

    Names follow the same convention as :func:`iter_quant_targets`: ``module_name`` for
    the ``weight`` parameter of ``nn.Linear`` / ``nn.Embedding`` and ``f"{name}.{pname}"``
    otherwise. These names lock the corresponding modules against re-quantization and QKV
    renormalization when merging with an existing checkpoint.
    """
    names: set[str] = set()
    for name, module in model.named_modules():
        for pname, param in module.named_parameters(recurse=False):
            if param is None:
                continue
            if isinstance(param, QuantTensor) or isinstance(getattr(param, "data", None), QuantTensor):
                if pname == "weight" and isinstance(module, (torch.nn.Linear, torch.nn.Embedding)):
                    names.add(name)
                else:
                    names.add(f"{name}.{pname}" if name else pname)
    return names


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

    skip_patterns = list(getattr(fresh_qcfg, "modules_to_not_convert", None) or [])
    component_embedding_name_set = set(component_embedding_names)

    def _iter_component_quant_targets(quant_cfg: OliveHfQuantizationConfig):
        for module, pname, full_name in iter_quant_targets(
            wrapper.model,
            quantize_lm_head=quant_cfg.lm_head,
            quantize_embeds=quant_cfg.embeds,
            quantize_moe=getattr(quant_cfg, "moe", False),
            quantize_vision=getattr(quant_cfg, "quantize_vision", False),
            skip_patterns=skip_patterns,
            extra_skip_modules=excluded_attn_inputs,
        ):
            root_name = _root_module_name(full_name, name_prefix)
            # When the slice spans more than the component (multi-path components slice
            # to a common ancestor), restrict quantization to the declared sub-trees.
            if not _is_in_component(root_name, component_source_paths):
                continue
            # For component-selected embedding quantization, only quantize embeddings that
            # belong to the selected component(s), not sibling embeddings under the slice.
            if pname == "weight" and isinstance(module, torch.nn.Embedding):
                if component_source_paths or isinstance(wrapper, _GenericComponentWrapper):
                    if root_name not in component_embedding_name_set:
                        continue
                elif embeds_name is not None and root_name != embeds_name:
                    continue
            yield module, pname, root_name

    fresh_names = {root_name for _, _, root_name in _iter_component_quant_targets(fresh_qcfg)}

    # Pre-existing quantized weights are immutable. If we're merging with an existing
    # checkpoint, build the final qcfg first (merge fresh into existing, then renormalize
    # QKV with already-quantized parameters locked) so that the quant_info we attach below
    # uses the same settings the on-disk fusion will require. Every parameter that is already
    # a ``QuantTensor`` after load is on-disk-immutable, including those that used the
    # existing config's defaults (no explicit override entry).
    on_disk_overrides: set[str] = set()
    already_quantized: set[str] = set()
    if existing_qcfg:
        on_disk_overrides = set((existing_qcfg.get("overrides") or {}).keys())
        already_quantized = {
            _root_module_name(name, name_prefix) for name in _collect_already_quantized_names(wrapper.model)
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
        merged["moe"] = merged.get("moe", False) or getattr(fresh_qcfg, "moe", False)
        merged["quantize_vision"] = merged.get("quantize_vision", False) or getattr(
            fresh_qcfg, "quantize_vision", False
        )
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

    for module, pname, root_name in _iter_component_quant_targets(qcfg):
        qargs = qcfg.get_qlinear_init_args(root_name)
        new_qargs[root_name] = qargs
        module._parameters[pname].quant_info = QuantInfo(quantizer=WeightQuantizer(**qargs))

    # Drop overrides for modules that won't be quantized this pass. Pre-existing (on-disk)
    # overrides are preserved verbatim since they describe already-quantized weights.
    # QKV-group overrides for modules excluded from this pass are not kept: when the
    # follow-up pass runs, the quantized members in the group will be locked and pull the
    # remaining members back into the shared config via ``normalize_qkv_quant_config``.
    for name in list(qcfg.overrides or {}):
        # ``re:`` keys aren't tied to a specific module, so leave them in place.
        if name.startswith("re:"):
            continue
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
        "moe": getattr(config, "moe", False),
        "quantize_vision": getattr(config, "quantize_vision", False),
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

    try:
        for data in get_calibration_dataset(model, data_config):
            try:
                wrapper.model(**tensor_data_to_device(data, device))
            except ValueError:
                # `store_input_hook` raises ValueError once it has captured the first layer's
                # inputs, deliberately aborting the forward pass early since the rest of the
                # model's computation isn't needed for calibration.
                pass
    finally:
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
    moe_session: MoeCalibrationSession | None = None,
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
        moe_session: Optional MoE calibration session. Required when any selected parameter
            lives on a fused MoE experts module: routing happens *inside* one experts-module
            forward call, so per-expert activations cannot be observed with an ordinary
            forward hook. The session intercepts the experts forward instead and records one
            independent Hessian per expert. Ordinary ``nn.Linear`` / ``nn.Embedding``
            targets keep using ``input_hook``.

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

    # Everything below runs inside try/finally: the experts-implementation swap, the
    # ``use_cache`` override, the forward hooks and the progress bar are all process-global
    # mutations that must be undone even when calibration raises part way through.
    pbar = None
    handles: list = []
    try:
        hidden_states, layer_args, layer_kwargs = get_layer_inputs_for_calibration(model, wrapper, data_config, device)
        if not hidden_states:
            raise ValueError("Calibration data is empty. Provide a valid data_config.")

        total_steps = wrapper.num_hidden_layers + (1 if include_lm_head else 0)
        pbar = tqdm(total=total_steps, desc="Processing layers...")

        if moe_session is not None:
            moe_session.start()

        layers_name = wrapper.get_layers(return_name=True)[1]
        for layer_idx, layer in enumerate(wrapper.get_layers(return_name=False)):
            pbar.set_postfix(module=f"layers.{layer_idx}", refresh=False)
            dense_modules, moe_modules = _split_quantizable_modules(layer)
            if moe_modules and moe_session is None:
                raise ValueError(
                    "Fused MoE expert parameters were selected for calibrated quantization but no "
                    "MoE calibration session was provided. This is an internal error."
                )
            handles = [module.register_forward_hook(input_hook) for module in dense_modules]

            record_ctx = (
                moe_session.record(moe_modules) if moe_session is not None and moe_modules else contextlib.nullcontext()
            )
            with record_ctx:
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
            handles = []

            if moe_session is not None:
                layer_name = f"{layers_name}.{layer_idx}"
                for experts in moe_modules:
                    moe_session.add_coverage(layer_name, experts)

            for module in [*dense_modules, *moe_modules]:
                process_module(module, device)

            if not update_before_process:
                # true-sequential: re-run the layer with the quantized weights so the next layer
                # sees realistic inputs. Recording is off here (the ``record`` context above has
                # exited), so Hessians are not double-counted.
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
            hidden_states = run_layer(
                wrapper.get_pre_head_layernorm(return_name=False), hidden_states, return_output=True
            )
            lm_head = wrapper.get_lm_head(return_name=False)
            pbar.set_postfix(module="lm_head", refresh=False)
            handles = [lm_head.register_forward_hook(input_hook)]
            run_layer(lm_head, hidden_states, return_output=True)
            for handle in handles:
                handle.remove()
            handles = []
            process_module(lm_head, device)
            pbar.update(1)
    finally:
        for handle in handles:
            handle.remove()
        if pbar is not None:
            pbar.close()
        try:
            if moe_session is not None:
                moe_session.finish()
        finally:
            # Always restore ``use_cache`` even if ``moe_session.finish()`` itself raises
            # (e.g. the experts-implementation restore call fails) -- these are two
            # independent process-global mutations and a failure in one must not leave the
            # other un-undone.
            if original_use_cache is not None:
                wrapper.model.config.use_cache = original_use_cache

    return device


def _module_weight_has_quant_info(module: torch.nn.Module) -> bool:
    """Return True if ``module.weight`` carries a ``quant_info`` attribute.

    Used by the layerwise discovery in :func:`run_layerwise_quantization`
    to locate ``nn.Linear`` modules selected for calibrated quantization
    without depending on a module-level attribute.
    """
    weight = getattr(module, "weight", None)
    return weight is not None and hasattr(weight, "quant_info")


def module_quant_info_param_names(module: torch.nn.Module) -> list[str]:
    """Return the names of ``module``'s direct parameters carrying ``quant_info``.

    Unlike :func:`_module_weight_has_quant_info` this does not hardcode the ``weight``
    attribute name, so it also finds fused-3D MoE expert parameters (``gate_up_proj`` /
    ``down_proj``), which :func:`prepare_model` selects via
    :func:`~olive.common.quant.selection.iter_quant_targets` but which were previously
    invisible to the layerwise discovery loop.
    """
    return [pname for pname, param in module._parameters.items() if param is not None and hasattr(param, "quant_info")]


def _split_quantizable_modules(
    layer: torch.nn.Module,
) -> tuple[list[torch.nn.Module], list[torch.nn.Module]]:
    """Split a layer's quantization targets into ordinary and fused-MoE modules.

    Returns ``(dense_modules, moe_modules)``:

    * ``dense_modules`` own a ``weight`` parameter with ``quant_info`` (``nn.Linear`` /
      ``nn.Embedding``) and are calibrated with an ordinary forward hook;
    * ``moe_modules`` own other ``quant_info``-carrying parameters -- fused-3D MoE experts
      weights (``gate_up_proj`` / ``down_proj``) -- whose per-expert activations are only
      observable from inside the experts forward.
    """
    dense_modules: list[torch.nn.Module] = []
    moe_modules: list[torch.nn.Module] = []
    for module in layer.modules():
        if _module_weight_has_quant_info(module):
            dense_modules.append(module)
        elif module_quant_info_param_names(module):
            moe_modules.append(module)
    return dense_modules, moe_modules


def _iter_quant_info_params(model: torch.nn.Module):
    """Yield ``(module, pname, param, quant_info)`` for every selected parameter."""
    for sub_module in model.modules():
        for pname in list(sub_module._parameters):
            param = sub_module._parameters.get(pname)
            if param is None:
                continue
            info = getattr(param, "quant_info", None)
            if info is None:
                continue
            yield sub_module, pname, param, info


def finalize(
    model: HfModelHandler,
    output_model_path: str,
    wrapper: ModelWrapper,
    quant_config: OliveHfQuantizationConfig,
    device: str,
    retie_word_embeddings: bool = False,
) -> HfModelHandler:
    """Finalize quantization by installing ``QuantTensor`` parameters in place.

    Walks every ``nn.Parameter`` whose tensor has a ``quant_info``
    attribute (set by :func:`prepare_model`), builds a ``QuantTensor``
    from the float tensor plus computed qparams, and installs it via
    :func:`install_quant_tensor_param` so that:

    * ``module.<pname>`` is an ``nn.Parameter(QuantTensor)`` whose
      dispatch still drives the original eager forward;
    * ``module.<pname>_qweight`` / ``_scales`` / ``_qzeros`` are plain
      buffers (aliasing the QuantTensor's inner tensors), so the model
      saves cleanly via ``save_pretrained`` / safetensors with no
      tensor subclass on disk.

    The same code path handles 2D linear/embedding weights and any
    higher-rank fused parameter (e.g. 3D MoE experts) — quantization
    is always along the last dim.
    """
    # Group selected params by their owning module so the module is
    # moved to ``device`` once even when it owns multiple quantized
    # parameters (fused-MoE experts modules carry two 3D tensors).
    by_module: dict[int, tuple[torch.nn.Module, list[tuple[str, torch.nn.Parameter, QuantInfo]]]] = {}
    for sub_module, pname, param, info in _iter_quant_info_params(wrapper.model):
        entry = by_module.setdefault(id(sub_module), (sub_module, []))
        entry[1].append((pname, param, info))

    for sub_module, params in by_module.values():
        sub_module.to(device)
        with torch.no_grad():
            built = []
            for pname, param, info in params:
                quantizer = info.quantizer
                qt = QuantTensor.from_float(
                    param.data.detach(),
                    bits=quantizer.bits,
                    symmetric=quantizer.symmetric,
                    group_size=quantizer.group_size,
                    scales=info.scales,
                    zero_points=info.zero_points,
                ).to("cpu")
                built.append((pname, qt))
        sub_module.to("cpu")
        for pname, qt in built:
            install_quant_tensor_param(sub_module, pname, qt)

    if retie_word_embeddings:
        tie_quant_word_embeddings(wrapper.model)
        quant_config.tie_word_embeddings = True

    if getattr(quant_config, "moe", False):
        logger.warning(
            "MoE weights have been quantized as 3D tensor parameters. The resulting checkpoint is "
            "save/load compatible via transformers but is not directly exportable to ONNX with the "
            "Olive ONNX conversion pass — consume it via the ORT GenAI model_builder or Mobius."
        )

    wrapper.model.quantization_method = quant_config.quant_method
    wrapper.model.config.quantization_config = quant_config

    # save the quantized model — state_dict hooks drop QuantTensor entries;
    # only plain ``<pname>_qweight`` / ``_scales`` / ``_qzeros`` buffers
    # are written to safetensors.
    #
    # Newer ``transformers`` versions (>=5.x) default ``save_pretrained`` to
    # ``save_original_format=True``, which for MoE architectures (e.g. Mixtral)
    # round-trips the on-disk state dict through a *legacy* per-expert
    # ``nn.Linear``-shaped layout (splitting the fused-3D ``experts.gate_up_proj``/
    # ``down_proj`` into ``experts.{i}.w1/w2/w3.weight`` and back). That
    # reshape/(un)fuse machinery assumes plain float weight tensors and silently
    # corrupts our quantized ``_scales``/``_qzeros`` buffers (their trailing
    # group-size dimension gets dropped), which crashes real forward passes after
    # a save/reload round-trip. Request the new, non-legacy on-disk format so the
    # fused-3D quantized buffers are written byte-for-byte as-is. Older
    # ``transformers`` versions don't accept this kwarg (they also don't have the
    # legacy-format conversion machinery, so there's nothing to opt out of).
    save_kwargs = {}
    if "save_original_format" in inspect.signature(wrapper.model.save_pretrained).parameters:
        save_kwargs["save_original_format"] = False
    wrapper.model.save_pretrained(output_model_path, **save_kwargs)
    model.save_metadata(output_model_path)

    return inherit_hf_from_hf(model, output_model_path, adapter_path=model.adapter_path)
