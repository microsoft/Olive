# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access
"""State-dict helpers for Olive's quantized weight representation.

Olive's quantization layout:

* The quantized state of a weight named ``<pname>`` (typically
  ``"weight"`` for ``nn.Linear``/``nn.Embedding``, or e.g.
  ``"gate_up_proj"`` for a fused-3D MoE expert tensor) is stored as
  plain buffers on the host module:

  * ``<pname>_qweight``  - packed uint8 tensor (always present)
  * ``<pname>_scales``   - per-group scales (always present)
  * ``<pname>_qzeros``   - per-group zero points (asymmetric only)

  This matches the suffix convention so on-disk safetensors keys are
  HF-loader friendly:  ``model.layers.0.mlp.gate_proj.weight_qweight``,
  ``...experts.gate_up_proj_qweight``, etc.

* At runtime, the host module's original parameter
  ``module._parameters[pname]`` becomes
  ``nn.Parameter(QuantTensor(...), requires_grad=False)``. The
  ``QuantTensor``'s inner ``qweight``/``scales``/``qzeros`` references
  alias the same Python tensor objects as the buffers above, so
  ``module.<pname>`` is a live view over the buffers and the original
  forward (e.g. ``F.linear``) dispatches through
  ``QuantTensor.__torch_function__``.

To keep save/load simple we only need a single state-dict save hook
(to suppress the QuantTensor parameter entry, since the buffers already
carry the data) plus a post-load helper that refreshes the QuantTensor
inner references after HF assigns freshly-loaded buffer tensors.
"""

from __future__ import annotations

import torch

_INSTALLED_FLAG = "_olive_quant_state_dict_hook_installed"

QWEIGHT_SUFFIX = "_qweight"
SCALES_SUFFIX = "_scales"
QZEROS_SUFFIX = "_qzeros"


def buffer_names(pname: str) -> tuple[str, str, str]:
    """Return the ``(qweight, scales, qzeros)`` buffer names for parameter ``pname``."""
    return f"{pname}{QWEIGHT_SUFFIX}", f"{pname}{SCALES_SUFFIX}", f"{pname}{QZEROS_SUFFIX}"


def _save_hook(module: torch.nn.Module, state_dict: dict, prefix: str, local_metadata: dict) -> None:
    """Drop ``QuantTensor`` parameter entries from ``state_dict``.

    Inner ``qweight``/``scales``/``qzeros`` tensors are already exposed
    as plain buffers on ``module`` and therefore appear in ``state_dict``
    under their own keys; the QuantTensor parameter entry is a redundant
    (and non-serialisable) duplicate.
    """
    # Local import to avoid a circular dependency at module-import time.
    from olive.common.quant.tensor import QuantTensor

    for pname in list(module._parameters):
        full_key = f"{prefix}{pname}"
        value = state_dict.get(full_key)
        if isinstance(value, QuantTensor):
            del state_dict[full_key]


def install_state_dict_hooks(module: torch.nn.Module) -> None:
    """Install Olive's state-dict save hook on ``module`` (idempotent)."""
    if getattr(module, _INSTALLED_FLAG, False):
        return
    module._register_state_dict_hook(_save_hook)
    setattr(module, _INSTALLED_FLAG, True)


def ensure_state_dict_hooks(model: torch.nn.Module) -> None:
    """Install the save hook on every submodule that hosts a QuantTensor parameter.

    Belt-and-suspenders for paths that may install a ``QuantTensor``
    parameter without going through :func:`install_quant_tensor_param`
    (e.g. retie helpers, future loaders). The hook is idempotent.
    """
    from olive.common.quant.tensor import QuantTensor

    for sub_module in model.modules():
        for param in sub_module._parameters.values():
            if param is None:
                continue
            if isinstance(param, QuantTensor) or isinstance(getattr(param, "data", None), QuantTensor):
                install_state_dict_hooks(sub_module)
                break


def install_quant_tensor_param(
    module: torch.nn.Module,
    pname: str,
    qt,  # QuantTensor
) -> None:
    """Install ``qt`` on ``module`` as ``<pname>`` plus aliased sibling buffers.

    Replaces ``module._parameters[pname]`` with ``nn.Parameter(qt)`` and
    registers ``<pname>_qweight``/``_scales``/(optionally) ``_qzeros``
    as buffers whose storage is the same as the QuantTensor's inner
    tensors. The state-dict save hook is installed as a side effect.
    """
    from olive.common.quant.tensor import QuantTensor

    if not isinstance(qt, QuantTensor):
        raise TypeError(f"Expected QuantTensor, got {type(qt).__name__}")

    qname, sname, zname = buffer_names(pname)

    # Detach existing buffers/parameters with the same names first so
    # ``register_buffer`` / parameter assignment is idempotent.
    for n in (qname, sname, zname):
        if n in module._buffers:
            del module._buffers[n]

    # ``nn.Parameter(qt, requires_grad=False)`` for a tensor subclass
    # returns the underlying QuantTensor instance directly (after a
    # ``detach()`` that goes through ``_apply_fn_to_data`` and produces
    # view-aliased inner tensors). See ``torch.nn.Parameter.__new__`` —
    # for a tensor subclass it returns ``data.detach().requires_grad_(...)``
    # which is the QuantTensor itself, not a wrapping Parameter. We
    # alias the host module's buffers to *that* instance's inner tensors
    # so save / refresh paths consistently read the same storage.
    param = torch.nn.Parameter(qt, requires_grad=False)
    module._parameters[pname] = param

    module.register_buffer(qname, param.qweight, persistent=True)
    module.register_buffer(sname, param.scales, persistent=True)
    if param.qzeros is not None:
        module.register_buffer(zname, param.qzeros, persistent=True)

    install_state_dict_hooks(module)


def get_quant_buffers(module: torch.nn.Module, pname: str):
    """Return ``(qweight, scales, qzeros)`` from ``module._buffers`` for ``pname``.

    Returns ``None`` when the module does not carry a complete set of quantized buffers
    (``qzeros`` is legitimately absent for symmetric quantization, ``qweight``/``scales``
    are not).
    """
    qname, sname, zname = buffer_names(pname)
    qweight = module._buffers.get(qname)
    scales = module._buffers.get(sname)
    if qweight is None or scales is None:
        return None
    return qweight, scales, module._buffers.get(zname)


def bind_quant_tensor_to_buffers(qt, module: torch.nn.Module, pname: str) -> bool:
    """Point ``qt``'s inner tensor refs at ``module``'s current ``<pname>_*`` buffers.

    Returns ``True`` when at least one of the module's buffer objects differed (by identity)
    from what ``qt`` referenced before, ``False`` when nothing changed or when ``module`` has
    no quantized buffers for ``pname``. Note this is deliberately *not* the "was this
    parameter loaded from a checkpoint" signal — that requires evidence for **every**
    mandatory buffer; see :func:`_all_buffers_replaced`.
    """
    buffers = get_quant_buffers(module, pname)
    if buffers is None:
        return False
    qweight, scales, qzeros = buffers
    changed = qt.qweight is not qweight or qt.scales is not scales or qt.qzeros is not qzeros
    qt.qweight = qweight
    qt.scales = scales
    qt.qzeros = qzeros
    return changed


def alias_module_buffers_to_quant_tensor(qt, module: torch.nn.Module, pname: str) -> None:
    """Re-point ``module``'s ``<pname>_*`` buffer dict entries at ``qt``'s inner tensors.

    Used to keep tied/aliased hosting modules (e.g. ``lm_head`` tied to
    ``model.embed_tokens``) pointing at the exact same buffer objects the shared
    ``QuantTensor`` uses, so ``state_dict()`` / save and live forward computation agree.
    """
    qname, sname, zname = buffer_names(pname)
    for name, tensor in ((qname, qt.qweight), (sname, qt.scales), (zname, qt.qzeros)):
        if name in module._buffers and tensor is not None:
            module._buffers[name] = tensor


def _full_name(prefix: str, name: str) -> str:
    return f"{prefix}.{name}" if prefix else name


def _collect_quant_hosts(module: torch.nn.Module):
    """Group every ``QuantTensor``-hosting site in ``module`` by QuantTensor object identity.

    Returns ``{id(qt): (qt, [(prefix, sub_module, pname), ...])}``. Tied weights install the
    very same ``QuantTensor`` object on two modules, so a single entry can have multiple
    hosting sites; sites are listed in ``named_modules()`` order.
    """
    from olive.common.quant.tensor import QuantTensor

    hosts: dict[int, tuple[QuantTensor, list[tuple[str, torch.nn.Module, str]]]] = {}
    for prefix, sub_module in module.named_modules():
        for pname, param in list(sub_module._parameters.items()):
            if param is None:
                continue
            # ``param`` itself is the QuantTensor instance stored on the module
            # (``nn.Parameter(qt)`` for a tensor subclass returns the underlying
            # QuantTensor — see ``torch.nn.Parameter.__new__``).
            qt = param if isinstance(param, QuantTensor) else param.data
            if not isinstance(qt, QuantTensor):
                continue
            if get_quant_buffers(sub_module, pname) is None:
                continue
            hosts.setdefault(id(qt), (qt, []))[1].append((prefix, sub_module, pname))
    return hosts


def _mandatory_buffer_names(qt, pname: str) -> tuple[str, ...]:
    """Return the buffer names that must be loaded for ``pname`` to count as fully loaded.

    ``qweight`` and ``scales`` always exist. ``qzeros`` only exists for asymmetric
    quantization — ``qt.qzeros`` (not ``qt.symmetric``) is authoritative, because it is
    what ``install_quant_tensor_param`` keys the buffer registration off. Requiring a
    ``_qzeros`` key/swap for a symmetric parameter that legitimately has no such buffer
    would fail-close on a perfectly complete checkpoint.
    """
    qname, sname, zname = buffer_names(pname)
    return (qname, sname, zname) if qt.qzeros is not None else (qname, sname)


def _all_keys_in_checkpoint(qt, prefix: str, pname: str, checkpoint_keys: set[str]) -> bool:
    """Whether *every* mandatory buffer of this site has a key in the checkpoint manifest.

    Checking only ``<pname>_qweight`` would call a truncated/corrupt checkpoint that carries
    e.g. ``_scales`` but not ``_qweight`` "loaded", clearing ``is_placeholder`` over
    zero-filled weights.
    """
    return all(_full_name(prefix, name) in checkpoint_keys for name in _mandatory_buffer_names(qt, pname))


def _all_buffers_replaced(qt, sub_module: torch.nn.Module, pname: str) -> bool:
    """Whether *every* mandatory buffer object of this site differs (by identity) from ``qt``'s.

    Only a real ``load_state_dict(..., assign=True)`` replaces the buffer objects in
    ``module._buffers``; a parameter that received nothing keeps the exact placeholder
    buffer objects ``install_quant_tensor_param`` registered. Requiring *all* of them to
    have been replaced (rather than any one) is what keeps a partial load — e.g. only
    ``_scales`` swapped while ``_qweight`` is still the all-zero placeholder — from being
    miscounted as a full load.
    """
    buffers = get_quant_buffers(sub_module, pname)
    if buffers is None:
        return False
    qweight, scales, qzeros = buffers
    if qt.qweight is qweight or qt.scales is scales:
        return False
    # A missing ``_qzeros`` buffer is not evidence of a load either, hence the ``is None`` arm.
    return not (qt.qzeros is not None and (qzeros is None or qt.qzeros is qzeros))


def _select_source_site(qt, sites, checkpoint_keys: set[str] | None) -> tuple[int, bool]:
    """Pick which hosting site's buffers hold the freshly-loaded checkpoint data.

    Returns ``(index_into_sites, was_loaded)``. ``was_loaded`` is ``True`` only when some
    site has evidence that *all* of its mandatory buffers (see
    :func:`_mandatory_buffer_names`) received real data — from the checkpoint's key
    manifest or, failing that, from buffer-object identity. Partial evidence (only some of
    the buffers) counts as not loaded, so the caller fails closed instead of clearing
    ``is_placeholder`` over placeholder storage.
    """
    if checkpoint_keys is not None:
        # Exact key-membership check against the checkpoint's own manifest: authoritative
        # regardless of whether the loader replaced the buffer object (``setattr``) or
        # mutated it in place (``.copy_()``). For tied weights only the source module's
        # keys are persisted, so this also picks the correct module out of the alias group.
        for index, (prefix, _, pname) in enumerate(sites):
            if _all_keys_in_checkpoint(qt, prefix, pname, checkpoint_keys):
                return index, True

    # Buffer-object identity. This is the only available signal when the checkpoint's key
    # set is unknown, and it is also a necessary *second* signal when the manifest is known:
    # ``checkpoint_keys`` holds the raw on-disk key names, but HF can remap them while
    # loading (e.g. ``save_original_format=True`` writes MoE experts as legacy per-expert
    # ``experts.{i}.w1.weight_qweight`` keys that the loader fuses back into
    # ``experts.gate_up_proj_qweight``), so a missing raw key does *not* by itself prove the
    # parameter went unloaded.
    for index, (_, sub_module, pname) in enumerate(sites):
        if _all_buffers_replaced(qt, sub_module, pname):
            return index, True
    return 0, False


def refresh_quant_tensor_refs(module: torch.nn.Module, checkpoint_keys: set[str] | None = None) -> None:
    """Re-point each ``QuantTensor`` parameter at the module's current buffers.

    HF's loader assigns freshly-loaded buffer tensors via
    ``module.load_state_dict({name: tensor}, assign=True)``, which
    replaces the buffer object in ``module._buffers``. Any
    ``QuantTensor`` parameter installed earlier would still reference
    the old (placeholder) storage, so we walk the parameters and re-bind
    each one's inner tensors to the current buffers.

    Each **unique** ``QuantTensor`` object is processed exactly once, keyed by object
    identity. Tied weights (``lm_head`` tied to ``model.embed_tokens``) host the *same*
    ``QuantTensor`` object on two modules: rebinding once per hosting module would be
    last-write-wins over ``named_modules()`` order, and the alias module's ``_buffers``
    entries are a one-time snapshot taken at tie time (HF's loader replaces the *source*
    module's buffer objects, not the alias's), so the losing write could silently bind the
    shared tensor to stale placeholder storage. This function is therefore the single
    source of truth for "which buffers does a QuantTensor actually reference": after it
    runs, every aliasing hosting module's ``_buffers`` entries are re-pointed at the same
    objects the shared ``QuantTensor`` references, so ``state_dict()`` / save and live
    forward computation agree without needing a separate repair pass.

    This is a no-op for modules with no QuantTensor parameters.

    Args:
        module: root module to walk.
        checkpoint_keys: when provided, the full set of tensor keys actually present
            in the checkpoint (e.g. read from the safetensors file headers). It is the
            preferred signal for picking the source site and for ``is_placeholder``: the
            full dotted name of *every* mandatory buffer of a parameter
            (``_qweight``/``_scales``, plus ``_qzeros`` for asymmetric quantization) is
            checked for membership directly, independent of *how* the loader wrote the
            value (``setattr`` replacement or an in-place ``.copy_()``). When ``None``
            (checkpoint format/files unknown), only the buffer-identity heuristic is used.

    Raises:
        RuntimeError: when ``checkpoint_keys`` is provided (so the checkpoint manifest is
            known) and a quantized parameter is still a placeholder that neither signal
            shows as *fully* loaded — i.e. the manifest is missing at least one of its
            mandatory buffer keys **and** at least one of its buffer objects was never
            replaced by the loader. Without this the model would silently hold (fully or
            partially) zero-filled placeholder weights. Fail-closed only applies when the
            manifest is known; with ``checkpoint_keys=None`` the permissive identity
            heuristic is preserved and nothing is raised — a partially loaded parameter
            simply keeps ``is_placeholder=True``.

    """
    missing: list[str] = []
    for qt, sites in _collect_quant_hosts(module).values():
        index, was_loaded = _select_source_site(qt, sites, checkpoint_keys)
        if not was_loaded and checkpoint_keys is not None and qt.is_placeholder:
            prefix, _, pname = sites[0]
            missing.append(_full_name(prefix, pname))
            continue

        src_prefix, src_module, src_pname = sites[index]
        bind_quant_tensor_to_buffers(qt, src_module, src_pname)
        for other_prefix, other_module, other_pname in sites:
            if other_prefix == src_prefix and other_pname == src_pname:
                continue
            alias_module_buffers_to_quant_tensor(qt, other_module, other_pname)

        if was_loaded:
            # Real checkpoint data is now bound to this parameter — clear the
            # placeholder lifecycle flag so in-place initializers on it raise instead
            # of silently no-oping (the no-op is only valid before real data is
            # loaded).
            qt.is_placeholder = False

    if missing:
        raise RuntimeError(
            "Quantized checkpoint is missing weights for: "
            + ", ".join(sorted(missing))
            + ". The model's quantization_config declares these parameters as quantized, but the "
            "checkpoint does not contain the complete set of `<pname>_qweight` / `_scales` "
            "(/ `_qzeros`) keys for them, so they would silently load as (partially) zero-filled "
            "placeholders. The checkpoint is incomplete or corrupt."
        )
