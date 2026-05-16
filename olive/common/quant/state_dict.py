# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
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


def refresh_quant_tensor_refs(module: torch.nn.Module) -> None:
    """Re-point each ``QuantTensor`` parameter at the module's current buffers.

    HF's loader assigns freshly-loaded buffer tensors via
    ``module.load_state_dict({name: tensor}, assign=True)``, which
    replaces the buffer object in ``module._buffers``. Any
    ``QuantTensor`` parameter installed earlier would still reference
    the old (placeholder) storage, so we walk the parameters and re-bind
    each one's inner tensors to the current buffers.

    This is a no-op for modules with no QuantTensor parameters.
    """
    from olive.common.quant.tensor import QuantTensor

    for sub_module in module.modules():
        for pname, param in list(sub_module._parameters.items()):
            # ``param`` itself is the QuantTensor instance stored on the
            # module (``nn.Parameter(qt)`` for a tensor subclass returns
            # the underlying QuantTensor — see ``torch.nn.Parameter.__new__``).
            if param is None or not (isinstance(param, QuantTensor) or isinstance(param.data, QuantTensor)):
                continue
            qname, sname, zname = buffer_names(pname)
            qweight = sub_module._buffers.get(qname)
            scales = sub_module._buffers.get(sname)
            qzeros = sub_module._buffers.get(zname)
            if qweight is None or scales is None:
                continue
            param.qweight = qweight
            param.scales = scales
            param.qzeros = qzeros
