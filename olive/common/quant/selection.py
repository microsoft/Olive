# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Quantization target selection.

Centralises the logic that walks a model once and decides which
parameters to quantize. Both Olive's HF quantizer (which installs
:class:`QuantTensor` placeholders before weight loading) and the
PyTorch RTN/GPTQ passes (which attach calibration metadata) consume
the same set of targets — only the per-target action differs.

Every target is a single ``nn.Parameter``, yielded as a
``(module, pname, full_name)`` tuple. ``full_name`` is the key used
for overrides / skip-pattern lookups (``module_name`` for ``"weight"``
on ``nn.Linear`` / ``nn.Embedding``; ``f"{module_name}.{pname}"``
otherwise). The selector makes no distinction between 2D linear /
embedding weights and 3D fused-MoE parameters — downstream code reads
the parameter's own shape and lets
:class:`~olive.common.quant.utils.WeightQuantizer` handle any rank
along the last dim.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch.nn as nn

from olive.common.hf.io_config.io_resolver import resolve_alias
from olive.common.quant.patterns import match_skip

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from olive.common.hf.wrapper import ModelWrapper


QuantTarget = tuple[nn.Module, str, str]
"""``(module, pname, full_name)`` for a single parameter selected for quantization."""


def _collect_experts(
    model: nn.Module,
    wrapper: ModelWrapper | None,
) -> list[tuple[nn.Module, str]]:
    """Return ``(experts_module, dotted_name)`` for every MoE layer."""
    if wrapper is None:
        return []
    out: list[tuple[nn.Module, str]] = []
    for lw in wrapper.get_layer_wrappers():
        experts, name = lw.get_experts(return_name=True)
        if experts is not None:
            out.append((experts, name))
    return out


def _collect_moe_routers(wrapper: ModelWrapper | None) -> list[nn.Module]:
    """Return the router module of every layer that also resolves an experts subtree.

    Routers decide which experts a token is sent to; quantizing them changes the routing
    decisions themselves, so they are kept in full precision. Most architectures wrap the
    router in a dedicated module (``MixtralTopKRouter``, ``GraniteMoeTopKRouter``, ...) that
    the ``nn.Linear``/``nn.Embedding`` walk never sees, but some -- e.g. Jamba, whose
    ``JambaSparseMoeBlock.router`` is a bare ``nn.Linear`` -- would otherwise be swept into
    the ordinary 2D walk. Excluding by *resolved module identity* (rather than by name
    pattern) covers both shapes.

    Only routers of layers with resolvable experts are excluded, so a dense layer that
    happens to own an attribute named ``gate`` is never silently skipped.
    """
    if wrapper is None:
        return []
    routers: list[nn.Module] = []
    for lw in wrapper.get_layer_wrappers():
        get_router = getattr(lw, "get_router", None)
        if get_router is None:
            continue
        router = get_router(return_name=False)
        if router is None or lw.get_experts(return_name=False) is None:
            continue
        routers.append(router)
    return routers


def _layers_missing_experts(wrapper: ModelWrapper | None) -> list[int]:
    """Return indices of layers that look structurally MoE but whose experts couldn't be resolved.

    A layer is only counted here when it *has a router* (``LayerWrapper.get_router()`` is not
    ``None``) — dense layers legitimately interleaved with MoE layers (e.g. DeepSeek's
    ``first_k_dense_replace``) have no router and are exempt, avoiding false positives on
    architectures with a mix of dense and MoE layers.
    """
    if wrapper is None:
        return []
    missing: list[int] = []
    for i, lw in enumerate(wrapper.get_layer_wrappers()):
        get_router = getattr(lw, "get_router", None)
        if get_router is None:
            # Test doubles / minimal wrappers without a get_router accessor cannot signal
            # "structurally MoE" — treat as unknown (not missing) rather than raising here.
            continue
        router = get_router(return_name=False)
        if router is None:
            continue
        experts = lw.get_experts(return_name=False)
        if experts is None:
            missing.append(i)
    return missing


# Canonical alias key for the MoE expert count; the candidate attribute paths (flat and
# nested, e.g. DBRX's ``ffn_config.moe_num_experts``) live in
# ``olive/assets/io_configs/defaults.yaml`` under ``aliases.num_experts`` so new
# architectures are added as data, not code.
_MOE_EXPERT_COUNT_ALIAS = "num_experts"

# Leaf attribute names used by the generic sub-config sweep below. Kept deliberately tight:
# a false positive here turns into a hard refusal to quantize.
_MOE_LEAF_ATTRS = ("num_local_experts", "num_experts", "n_routed_experts", "moe_num_experts")

# Nested sub-config names to sweep in addition to whatever HF declares in
# ``type(config).sub_configs``.
_EXTRA_SUB_CONFIG_NAMES = ("text_config", "thinker_config", "ffn_config", "decoder_config", "llm_config")

_MAX_SUB_CONFIG_DEPTH = 3


def _positive_int(value) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value > 0


def _sub_config_indicates_moe(config, depth: int) -> bool:
    """Bounded sweep of nested sub-configs for a positive expert-count attribute.

    Defense-in-depth for nested-MoE architectures not yet enumerated in
    ``aliases.num_experts``. Uses HF's own ``sub_configs`` declaration where available.
    """
    if depth <= 0 or config is None:
        return False
    names = list(getattr(type(config), "sub_configs", None) or ())
    names += [n for n in _EXTRA_SUB_CONFIG_NAMES if n not in names]
    for name in names:
        sub = getattr(config, name, None)
        if sub is None or isinstance(sub, (str, int, float, bool, list, tuple)):
            continue
        # ``sub`` may still be a raw dict if the config's ``post_init`` has not run.
        getter = sub.get if isinstance(sub, dict) else (lambda a, _s=sub: getattr(_s, a, None))
        if any(_positive_int(getter(attr)) for attr in _MOE_LEAF_ATTRS):
            return True
        if not isinstance(sub, dict) and _sub_config_indicates_moe(sub, depth - 1):
            return True
    return False


def _config_indicates_moe(model: nn.Module) -> bool:
    """Best-effort detection of an MoE architecture from the model config.

    Returns ``True`` only when a known MoE expert-count attribute is present and positive,
    either at the top level or on a nested sub-config (e.g. DBRX's
    ``config.ffn_config.moe_num_experts``). Used to fail closed when the experts subtree
    cannot be resolved.
    """
    config = getattr(model, "config", None)
    if config is None:
        return False
    if _positive_int(resolve_alias(config, _MOE_EXPERT_COUNT_ALIAS)):
        return True
    return _sub_config_indicates_moe(config, _MAX_SUB_CONFIG_DEPTH)


def iter_quant_targets(
    model: nn.Module,
    *,
    quantize_lm_head: bool,
    quantize_embeds: bool,
    quantize_moe: bool,
    skip_patterns: Iterable[str] = (),
    extra_skip_modules: Iterable[nn.Module] = (),
    skip_already_quantized: bool = True,
) -> Iterator[QuantTarget]:
    """Walk ``model`` once and yield every parameter selected for quantization.

    Yielded parameters are:

    * ``nn.Linear.weight`` and ``nn.Embedding.weight`` (2D), and
    * direct ``nn.Parameter`` attributes on each experts module
      (typically 3D fused-MoE weights), when ``quantize_moe=True``.

    Selection rules (first matching skip wins):

    * ``extra_skip_modules`` (caller-supplied set, e.g. attention
      inputs excluded by GPTQ) skips the module by identity.
    * ``quantize_lm_head=False`` skips the output embedding module.
    * ``quantize_embeds=False`` skips every ``nn.Embedding`` module.
      ``quantize_embeds=True`` targets only ``model.get_input_embeddings()``
      when resolvable (symmetric with ``lm_head``'s precise targeting of
      ``get_output_embeddings()``); falls back to every ``nn.Embedding``
      when the accessor is unavailable (e.g. non-HF synthetic fixtures).
    * ``quantize_moe=False`` skips every ``nn.Module`` under any
      experts subtree — this both leaves fused parameters alone *and*
      prevents silently quantizing per-expert ``nn.Linear``s inside
      ``ModuleList(Expert)`` blocks.
    * the router module of every MoE layer is always skipped (routers
      stay in full precision), including bare ``nn.Linear`` routers such
      as Jamba's.
    * ``skip_patterns`` matches the parameter's ``full_name`` via the
      shared HF-style substring / ``re:``-prefixed regex matcher.
    * When ``skip_already_quantized=True`` (default), parameters whose
      underlying tensor is already a :class:`QuantTensor` are skipped
      (idempotent re-runs).
    """
    from olive.common.hf.wrapper import ModelWrapper
    from olive.common.quant.tensor import QuantTensor

    try:
        wrapper = ModelWrapper.from_model(model)
    except Exception:  # pylint: disable=broad-except
        # Not every model is wrappable (e.g., random test fixtures).
        # Without the wrapper we cannot honour MoE / lm_head / embeds
        # category flags; fall back to the unfiltered 2D walk.
        wrapper = None

    lm_head_module: nn.Module | None = None
    if hasattr(model, "get_output_embeddings"):
        lm_head_module = model.get_output_embeddings()

    # Precise input-embedding target, symmetric with ``lm_head_module`` above. When
    # available, ``quantize_embeds=True`` targets *only* this module (matching the
    # documented "input embeddings" contract) rather than every ``nn.Embedding`` (which
    # would also sweep in positional / token-type tables). Fallback: when
    # ``get_input_embeddings`` is unavailable or returns ``None`` (e.g. synthetic test
    # fixtures without a full HF model), retain the broad "all nn.Embedding" behavior.
    input_embeds_module: nn.Module | None = None
    if hasattr(model, "get_input_embeddings"):
        input_embeds_module = model.get_input_embeddings()

    expert_modules = _collect_experts(model, wrapper)
    expert_module_ids = {id(m) for m, _ in expert_modules}

    # Fail-closed: if the model/config advertises an MoE architecture but we could not
    # resolve any experts subtree, refuse to walk. Silently falling through to the plain
    # 2D walk would (a) leave fused expert weights at full precision and (b) — worse —
    # quantize every ``nn.Linear`` under an unrecognized ``ModuleList`` experts subtree
    # even when ``quantize_moe=False``, reproducing the exact bug the ``moe`` flag fixes.
    # Raising here (before the generator yields any target) guarantees no parameter is
    # modified before the error surfaces.
    if not expert_modules and _config_indicates_moe(model):
        raise ValueError(
            "Model config indicates a Mixture-of-Experts architecture, but Olive could not "
            "locate its experts subtree (LayerWrapper.get_experts() returned nothing for every "
            "layer). This architecture is not yet supported by Olive's MoE-aware quantization "
            "walk. Refusing to quantize to avoid silently mis-handling the experts. Add the "
            "architecture's experts/router names to LayerWrapper.EXPERTS/ROUTER, or exclude the "
            "experts explicitly via modules_to_not_convert."
        )

    # Fail-closed, per-layer: even when *some* layers resolve experts, a layer that is
    # structurally MoE (it has a resolvable router/gate) but whose experts subtree failed to
    # resolve is a discovery failure for that layer, not a legitimately expert-free dense
    # layer. Architectures that legitimately interleave dense layers with MoE layers (e.g.
    # DeepSeek's ``first_k_dense_replace``) have no router on the dense layers, so they are
    # exempt and do not trip this guard.
    missing_layers = _layers_missing_experts(wrapper)
    if missing_layers:
        total_layers = len(wrapper.get_layer_wrappers()) if wrapper is not None else 0
        raise ValueError(
            "Olive detected a router/gate on "
            f"{len(missing_layers)} of {total_layers} decoder layers (indices "
            f"{missing_layers}) but could not resolve their experts subtree "
            "(LayerWrapper.get_experts() returned nothing). This looks like a partially "
            "supported Mixture-of-Experts architecture. Refusing to quantize to avoid "
            "silently leaving those layers' experts unquantized (or misclassifying their "
            "sub-modules) with moe=True. Add the architecture's experts/router names to "
            "LayerWrapper.EXPERTS/ROUTER, or exclude the affected layers explicitly via "
            "modules_to_not_convert."
        )

    # ID-based skip set for fast identity checks during the named_modules walk.
    skip_ids: set[int] = {id(m) for m in extra_skip_modules}
    if not quantize_lm_head and lm_head_module is not None:
        skip_ids.add(id(lm_head_module))
    # Routers stay full precision regardless of ``quantize_moe`` -- see
    # :func:`_collect_moe_routers`.
    for router in _collect_moe_routers(wrapper):
        for sub in router.modules():
            skip_ids.add(id(sub))
    if not quantize_moe:
        for experts, _ in expert_modules:
            for sub in experts.modules():
                skip_ids.add(id(sub))

    patterns = list(skip_patterns or ())

    def _is_skipped(module: nn.Module, full_name: str) -> bool:
        if id(module) in skip_ids:
            return True
        return bool(patterns) and match_skip(full_name, patterns)

    def _is_already_quantized(param) -> bool:
        return skip_already_quantized and (isinstance(param, QuantTensor) or isinstance(param.data, QuantTensor))

    for name, module in model.named_modules():
        # nn.Linear / nn.Embedding ``weight`` — legacy override-key
        # convention: full_name == module_name. When ``quantize_embeds``
        # is False every ``nn.Embedding`` is skipped (positional /
        # token-type / etc.) — this closes the loophole of an
        # unintended embedding sneaking through. When ``quantize_embeds``
        # is True and ``model.get_input_embeddings()`` is resolvable,
        # only that precise module is targeted (symmetric with
        # ``lm_head``'s precise targeting of ``get_output_embeddings()``);
        # otherwise (no HF accessor available) every ``nn.Embedding`` is
        # targeted, matching the previous broad behavior for non-HF
        # synthetic fixtures.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if isinstance(module, nn.Embedding):
                if not quantize_embeds:
                    continue
                if input_embeds_module is not None and module is not input_embeds_module:
                    continue
            if _is_skipped(module, name):
                continue
            weight = module.weight
            if weight is None or _is_already_quantized(weight):
                continue
            yield module, "weight", name
            continue

        # Fused-MoE pass: direct parameters on experts modules. Only 3D fused expert
        # *weight* tensors are quantization targets. Requiring ``dim() == 3`` (rather than
        # ``dim() in (2, 3)``) structurally excludes 2D non-weight params such as gpt-oss's
        # ``gate_up_proj_bias`` / ``down_proj_bias``, which must stay full precision.
        if not quantize_moe or id(module) not in expert_module_ids:
            continue
        for pname, param in module.named_parameters(recurse=False):
            if param is None or param.dim() != 3 or _is_already_quantized(param):
                continue
            full_name = f"{name}.{pname}" if name else pname
            if _is_skipped(module, full_name):
                continue
            yield module, pname, full_name
