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
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
import torch.nn as nn

from olive.common.quant.patterns import match_skip

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator
    from typing import Callable

    from olive.common.hf.wrapper import ModelWrapper


@dataclass
class QuantTarget:
    """A single parameter selected for quantization.

    Attributes:
        module: The owning ``nn.Module``.
        module_name: Dotted name of ``module`` relative to the model root.
        param_name: Name of the parameter on ``module`` (e.g. ``"weight"``
            for ``nn.Linear``/``nn.Embedding``; an arbitrary 3D
            parameter name for a fused-experts module).
        full_name: ``f"{module_name}.{param_name}"`` (with an exception
            for ``param_name == "weight"`` on ``nn.Linear``/``nn.Embedding``,
            in which case ``full_name == module_name`` — matching how
            overrides and skip patterns have always been keyed).
        kind: ``"linear"`` for ``nn.Linear``, ``"embedding"`` for
            ``nn.Embedding``, ``"fused_experts"`` for a 3D parameter on
            an experts module.
        shape: Logical (dequantized) shape of the parameter.
        dtype: Original parameter dtype.
        device: Original parameter device.

    """

    module: nn.Module
    module_name: str
    param_name: str
    full_name: str
    kind: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


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


def iter_quant_targets(
    model: nn.Module,
    *,
    quantize_lm_head: bool,
    quantize_embeds: bool,
    quantize_moe: bool,
    skip_patterns: Iterable[str] = (),
    extra_skip_modules: Iterable[nn.Module] = (),
    skip_already_quantized: bool = True,
    consider_linears: bool = True,
    consider_embeddings: bool = True,
) -> Iterator[QuantTarget]:
    """Walk ``model`` once and yield every parameter selected for quantization.

    Selection rules (first matching skip wins):

    * ``extra_skip_modules`` (caller-supplied set, e.g. attention inputs
      excluded by GPTQ) skips the module by identity.
    * ``quantize_lm_head=False`` skips the output embedding module.
    * ``quantize_embeds=False`` skips the input embedding module.
    * ``quantize_moe=False`` skips every ``nn.Module`` under any experts
      subtree (this both leaves fused-3D experts alone and prevents
      silently quantizing per-expert ``nn.Linear``s inside
      ``ModuleList(Expert)`` blocks).
    * ``skip_patterns`` matches the parameter's ``full_name`` via the
      shared HF-style substring / ``re:``-prefixed regex matcher.
    * When ``skip_already_quantized=True``, weights that are already a
      :class:`QuantTensor` are skipped (idempotent re-runs).

    When ``quantize_moe=True`` and an experts module exposes a 3D
    ``nn.Parameter`` (fused experts), that parameter is yielded as a
    ``"fused_experts"`` target. Per-expert 2D ``nn.Linear``s inside an
    ``nn.ModuleList`` experts wrapper continue to come through the
    regular linear walk.
    """
    from olive.common.hf.wrapper import ModelWrapper
    from olive.common.quant.tensor import QuantTensor

    try:
        wrapper = ModelWrapper.from_model(model)
    except Exception:  # pylint: disable=broad-except
        # Not every model is wrappable (e.g., random sklearn-like
        # test fixtures). Without the wrapper we cannot honour MoE /
        # lm_head / embeds category flags; fall back to the unfiltered
        # 2D walk.
        wrapper = None

    lm_head_module: nn.Module | None = None
    embed_module: nn.Module | None = None
    if hasattr(model, "get_output_embeddings"):
        lm_head_module = model.get_output_embeddings()
    if hasattr(model, "get_input_embeddings"):
        embed_module = model.get_input_embeddings()

    expert_modules = _collect_experts(model, wrapper)

    # ID-based skip set for fast identity checks during the named_modules walk.
    skip_ids: set[int] = {id(m) for m in extra_skip_modules}
    if not quantize_lm_head and lm_head_module is not None:
        skip_ids.add(id(lm_head_module))
    if not quantize_embeds and embed_module is not None:
        skip_ids.add(id(embed_module))
    if not quantize_moe:
        for experts, _ in expert_modules:
            for sub in experts.modules():
                skip_ids.add(id(sub))

    patterns = list(skip_patterns or ())

    def _is_skipped(module: nn.Module, full_name: str) -> bool:
        if id(module) in skip_ids:
            return True
        return bool(patterns) and match_skip(full_name, patterns)

    # 2D pass: every nn.Linear / nn.Embedding under the model.
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and consider_linears:
            kind = "linear"
        elif isinstance(module, nn.Embedding) and consider_embeddings:
            kind = "embedding"
        else:
            continue
        if _is_skipped(module, name):
            continue
        weight = module.weight
        if skip_already_quantized and isinstance(weight, QuantTensor):
            continue
        if weight is None:
            continue
        yield QuantTarget(
            module=module,
            module_name=name,
            param_name="weight",
            full_name=name,
            kind=kind,
            shape=tuple(weight.shape),
            dtype=weight.dtype,
            device=weight.device,
        )

    # 3D pass: fused-experts modules only when MoE quantization is requested.
    if not quantize_moe:
        return
    for experts_module, experts_name in expert_modules:
        for pname, param in experts_module.named_parameters(recurse=False):
            if param is None or param.dim() != 3:
                continue
            if skip_already_quantized and isinstance(param.data, QuantTensor):
                continue
            full_name = f"{experts_name}.{pname}" if experts_name else pname
            if _is_skipped(experts_module, full_name):
                continue
            yield QuantTarget(
                module=experts_module,
                module_name=experts_name,
                param_name=pname,
                full_name=full_name,
                kind="fused_experts",
                shape=tuple(param.shape),
                dtype=param.dtype,
                device=param.device,
            )


def for_each_target(
    model: nn.Module,
    handler: Callable[[QuantTarget], None],
    **selection_kwargs,
) -> list[QuantTarget]:
    """Run ``handler`` once per target and return the materialised list."""
    targets = list(iter_quant_targets(model, **selection_kwargs))
    for t in targets:
        handler(t)
    return targets
