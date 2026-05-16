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
    * ``quantize_embeds=False`` skips the input embedding module.
    * ``quantize_moe=False`` skips every ``nn.Module`` under any
      experts subtree — this both leaves fused parameters alone *and*
      prevents silently quantizing per-expert ``nn.Linear``s inside
      ``ModuleList(Expert)`` blocks.
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
    embed_module: nn.Module | None = None
    if hasattr(model, "get_output_embeddings"):
        lm_head_module = model.get_output_embeddings()
    if hasattr(model, "get_input_embeddings"):
        embed_module = model.get_input_embeddings()

    expert_modules = _collect_experts(model, wrapper)
    expert_module_ids = {id(m) for m, _ in expert_modules}

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

    def _is_already_quantized(param) -> bool:
        return skip_already_quantized and (isinstance(param, QuantTensor) or isinstance(param.data, QuantTensor))

    for name, module in model.named_modules():
        # nn.Linear / nn.Embedding ``weight`` — legacy override-key
        # convention: full_name == module_name.
        if isinstance(module, (nn.Linear, nn.Embedding)):
            if _is_skipped(module, name):
                continue
            weight = module.weight
            if weight is None or _is_already_quantized(weight):
                continue
            yield module, "weight", name
            continue

        # Fused-MoE pass: direct parameters on experts modules.
        if not quantize_moe or id(module) not in expert_module_ids:
            continue
        for pname, param in module.named_parameters(recurse=False):
            if param is None or _is_already_quantized(param):
                continue
            full_name = f"{name}.{pname}" if name else pname
            if _is_skipped(module, full_name):
                continue
            yield module, pname, full_name
