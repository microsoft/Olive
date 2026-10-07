# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Shared layout-safety checks for fused MoE expert quantization."""

from __future__ import annotations

import torch


class MoeSupportError(ValueError):
    """Raised when an MoE quantization layout cannot be proven safe to quantize."""


def check_moe_layout_support(
    experts_modules: list[torch.nn.Module],
    *,
    model_type: str,
    operation: str,
) -> None:
    """Fail closed unless every fused-experts module reports a K-last layout.

    The quantizers using this check group along each weight tensor's last dimension.
    Consequently, only fused weights stored ``(num_experts, out_features, in_features)``
    are safe: the input/contraction dimension K must be last.

    ``is_transposed`` is a boolean attribute exposed by ``transformers``'s
    ``use_experts_implementation`` decorator on fused-experts modules (``False`` for K-last
    architectures such as Mixtral/Qwen3-MoE/DeepSeek-V3/etc., ``True`` for architectures such
    as gpt-oss). It is assigned once, from the decorator's own argument, inside the wrapped
    ``__init__`` -- not derived from ``config`` -- so no checkpoint or config field can
    influence it for a standard (non-``trust_remote_code``) experts implementation. Its
    absence indicates either an undecorated experts implementation (e.g. an older
    ``transformers`` release, or an architecture such as llama4/aria that has not adopted the
    fused-experts decorator) or an unrecognized implementation, so it is treated as unsafe.

    Classic per-expert ``nn.ModuleList`` experts (e.g. PhiMoE, DeepSeek-V3, or Mixtral on
    older ``transformers`` releases without the fused-experts refactor) are exempt only when
    the ``ModuleList`` itself owns no direct 3D parameter: Olive's quantizer selection only
    ever groups a fused-experts module's *own* 3D parameters, and a plain ``ModuleList``
    container cannot have one (each per-expert child is a plain 2D ``nn.Linear``, whose
    weight is unconditionally ``(out, in)`` with K last). A ``ModuleList`` that does carry a
    direct 3D parameter falls through to the normal ``is_transposed`` check below, which
    rejects it (such a parameter has no ``is_transposed`` metadata to trust).

    Note: this trusts ``is_transposed`` as reported by the resolved experts class. A custom
    experts implementation loaded via ``trust_remote_code`` that misreports its own layout
    (e.g. sets ``is_transposed=False`` while actually storing transposed weights) is not
    caught here; that is treated as user-introduced misuse of an explicitly opted-in trust
    boundary, not a layout Olive can independently verify.

    Args:
        experts_modules: The fused-experts (or classic ``nn.ModuleList``) modules discovered
            for this model, one per MoE decoder layer.
        model_type: The resolved HF ``config.model_type`` of the model being quantized. Not
            used to decide the layout (that is entirely driven by ``is_transposed``); included
            only for diagnostic context in error messages.
        operation: Human-readable label for the calling pass/operation, prefixed onto error
            messages. Purely cosmetic: it does not affect validation logic.

    Raises:
        MoeSupportError: If a fused-experts module's ``is_transposed`` attribute is missing,
            not a ``bool``, or ``True``.

    """
    for experts in experts_modules:
        if isinstance(experts, torch.nn.ModuleList) and not any(
            param.dim() == 3 for param in experts.parameters(recurse=False)
        ):
            # No direct 3D parameter: Olive's quantizer selection never groups anything on
            # this module itself, only on its per-expert nn.Linear children (K is
            # unconditionally last for those), so there is no transposed-layout risk here.
            continue

        actual_class = type(experts).__name__
        is_transposed = getattr(experts, "is_transposed", None)
        if not isinstance(is_transposed, bool):
            raise MoeSupportError(
                f"{operation} cannot verify the layout of experts module '{actual_class}' "
                f"(model_type='{model_type}') because attribute 'is_transposed' is missing "
                "or not a boolean. This typically means an experts implementation that has "
                "not adopted the fused-experts decorator (e.g. an older transformers release, "
                "or an architecture such as llama4/aria), or an unrecognized implementation. "
                "Quantization is refused rather than risking grouping along the wrong "
                "dimension. Re-run with moe=False so the experts stay in full precision."
            )
        if is_transposed:
            raise MoeSupportError(
                f"{operation} refuses experts module '{actual_class}' (model_type='{model_type}') "
                "because it reports a transposed fused-weight layout (E, K, OUT), while "
                "last-dimension grouping requires (E, OUT, K) with K last. Architectures such "
                "as gpt-oss store this transposed layout. Re-run with moe=False so the "
                "experts stay in full precision."
            )
