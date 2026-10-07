# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=protected-access,super-init-not-called,redefined-builtin,not-callable
"""QuantTensor — wrapper ``torch.Tensor`` subclass for weight-quantized parameters.

It stores quantization buffers (``qweight``, ``scales``, ``qzeros``) but presents the
shape / dtype / device of the dequantized full-precision weight.

Design notes:

* The class is a **wrapper** subclass (``_make_wrapper_subclass``) — it
  carries no real storage of its own, so the dense FP weight is never
  materialized in memory; only the packed buffers are allocated.
* ``F.linear`` and ``F.embedding`` are dispatched via
  ``__torch_function__``:
  - Eager: unpack + dequantize on the fly and forward to the dense op.
  - Under ``torch.onnx.is_in_onnx_export()``: raise — Olive's ONNX
    conversion pass swaps any ``nn.Linear`` / ``nn.Embedding`` whose
    weight is a ``QuantTensor`` for the existing exportable
    ``QuantLinearNbit`` / ``QuantEmbeddingNbit`` ``nn.Module``s (see
    ``olive/common/hf/quant.py``) *before* the tracer ever inspects
    the parameter. This keeps the legacy ``com.microsoft::MatMulNBits`` /
    ``com.microsoft::GatherBlockQuantized`` symbolic emission intact.
* All other ops (including ``model.to(dtype/device)``, ``.detach()``,
  ``.contiguous()``, ``.clone()``) are routed through ``_apply_fn_to_data``
  via ``__torch_dispatch__`` so the inner buffers move with the wrapper.
* For 3D fused MoE experts (``(num_experts, out, in)``) the same buffers
  carry an additional leading dim. ``__getitem__`` / ``index_select`` on
  the leading dim return a 2D ``QuantTensor`` (so per-expert
  ``F.linear(current_state, weight[expert_idx])`` continues to dispatch
  through the same code path).
"""

from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn.functional as F

from olive.common.quant.utils import (
    WeightQuantizer,
    pack_to_uint8,
    unpack_from_uint8,
)

__all__ = ["QuantTensor", "implements"]


_TORCH_FN_TABLE: dict[Callable, Callable] = {}

# In-place random/constant weight-initializer ops that ``PreTrainedModel._initialize_weights``
# (and similar module-init code paths) may call on a freshly-installed *placeholder*
# QuantTensor parameter before its real buffers are filled in from a checkpoint. Their
# numeric effect is immediately discarded once the checkpoint's ``<pname>_qweight`` /
# ``_scales`` / ``_qzeros`` buffers are loaded, so treating them as no-ops (rather than
# raising, or worse, silently dequantizing a 3D expert tensor just to throw the result away)
# is both safe and necessary for HF model loading to succeed for MoE / fused-3D targets.
# The no-op only applies while ``QuantTensor.is_placeholder`` is True (see
# ``__torch_dispatch__`` below); on a real (already-quantized) QuantTensor these ops raise
# instead, since silently no-oping there would discard real data.
_NOOP_INIT_OPS: set[Callable] = {
    torch.ops.aten.normal_.default,
    torch.ops.aten.uniform_.default,
    torch.ops.aten.zero_.default,
    torch.ops.aten.fill_.Scalar,
    torch.ops.aten.fill_.Tensor,
}


def implements(*torch_fns: Callable) -> Callable[[Callable], Callable]:
    """Register a torch-function override for ``QuantTensor``."""

    def decorator(fn: Callable) -> Callable:
        for torch_fn in torch_fns:
            _TORCH_FN_TABLE[torch_fn] = fn
        return fn

    return decorator


def _midq(bits: int) -> int:
    return 1 << (bits - 1)


# Central message for the ONNX-export rejection of 3D fused-MoE QuantTensors. Olive's MoE
# quantization is storage-only; ONNX export of the experts is delegated to Mobius / ORT
# GenAI ModelBuilder. This must be raised for *every* unsupported 3D op that would otherwise
# reach a dequantizing fallback (not only matmul/bmm), so ONNX never silently emits a dense
# expert graph.
_MOE_ONNX_EXPORT_MSG = (
    "Olive's MoE quantization is storage-only: ONNX export of 3D fused-expert QuantTensors "
    "is not supported. Non-MoE parts export through the MatMulNBits / GatherBlockQuantized "
    "path; use Mobius / ORT GenAI ModelBuilder to emit com.microsoft.QMoE (or a per-expert "
    "MatMulNBits loop) for the experts."
)


def _zero_points_or_default(weight: QuantTensor) -> torch.Tensor:
    """Unpack zero_points or return a tensor full of the symmetric mid-q value."""
    if weight.qzeros is not None:
        return unpack_from_uint8(weight.qzeros, weight.bits, tuple(weight.scales.shape)).to(torch.int32)
    return torch.full(weight.scales.shape, _midq(weight.bits), dtype=torch.int32, device=weight.scales.device)


def _dequantize(weight: QuantTensor) -> torch.Tensor:
    """Unpack + dequantize ``weight`` into a dense tensor of ``weight.dtype``."""
    if weight.dim() not in (2, 3):
        raise NotImplementedError(f"QuantTensor only supports 2D / 3D layouts, got {weight.dim()}D")

    quantizer = WeightQuantizer(
        bits=weight.bits, symmetric=weight.symmetric, group_size=weight.group_size, signed=False
    )
    qw = unpack_from_uint8(weight.qweight, weight.bits, tuple(weight.shape))
    zp = _zero_points_or_default(weight)
    return quantizer.dequantize(qw, weight.scales, zp).to(weight.dtype)


class QuantTensor(torch.Tensor):
    """A weight-quantized tensor.

    Holds:
        qweight: ``torch.uint8`` packed quantized values along the last
            dim. Shape ``(*, math.ceil(in_features * bits / 8))``.
        scales:  per-group scales, dtype matches the dequantized dtype.
        qzeros:  ``torch.uint8`` packed zero-points, or ``None`` for
                 symmetric quantization.

    Attributes (non-tensor):
        bits, group_size, symmetric.

    The shape / dtype / device exposed via the wrapper subclass are
    those of the **dequantized** weight, so the host ``nn.Linear`` /
    ``nn.Embedding`` continues to see the right metadata.
    """

    qweight: torch.Tensor
    scales: torch.Tensor
    qzeros: torch.Tensor | None
    bits: int
    group_size: int
    symmetric: bool
    is_placeholder: bool

    @staticmethod
    def __new__(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        is_placeholder: bool = False,
    ) -> QuantTensor:
        return torch.Tensor._make_wrapper_subclass(  # type: ignore[attr-defined]
            cls,
            tuple(shape),
            dtype=dtype,
            device=qweight.device,
            requires_grad=False,
        )

    def __init__(
        self,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        shape: torch.Size | tuple[int, ...],
        dtype: torch.dtype,
        is_placeholder: bool = False,
    ) -> None:
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)
        # Lifecycle marker. True only for the zero-filled parameter installed by
        # ``OliveHfQuantizer._process_model_before_weight_loading`` *before* the checkpoint's
        # ``<pname>_qweight`` / ``_scales`` / ``_qzeros`` buffers are loaded. It is cleared by
        # ``refresh_quant_tensor_refs`` once real data is bound. In-place weight initializers
        # (``nn.init.normal_`` & friends, called by HF's ``PreTrainedModel._initialize_weights``)
        # are no-ops *only* while this is True; on a real quantized tensor they raise.
        self.is_placeholder = bool(is_placeholder)

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_float(
        cls,
        weight: torch.Tensor,
        bits: int = 4,
        symmetric: bool = True,
        group_size: int = -1,
        scales: torch.Tensor | None = None,
        zero_points: torch.Tensor | None = None,
    ) -> QuantTensor:
        """Quantize a 2D or 3D FP weight tensor and produce a ``QuantTensor``.

        Quantization is along the last dim — for 3D fused MoE weights of
        shape ``(num_experts, out, in)`` each ``(out, in)`` slice gets
        its own per-group scales / zero-points along ``in``, with no
        explicit leading-dim loop.
        """
        if weight.dim() not in (2, 3):
            raise ValueError(f"QuantTensor only supports 2D and 3D weights, got shape {tuple(weight.shape)}")

        quantizer = WeightQuantizer(bits=bits, symmetric=symmetric, group_size=group_size, signed=False)
        qparam_shape = quantizer.get_qparam_shape(tuple(weight.shape))

        if scales is None or zero_points is None:
            scales, zero_points = quantizer.find_qparams(weight)
        else:
            scales = scales.to(weight.device).to(weight.dtype).reshape(qparam_shape)
            zero_points = zero_points.to(weight.device).to(torch.int32).reshape(qparam_shape)

        qweight_int = quantizer.quantize(weight, scales, zero_points)
        qweight_packed = pack_to_uint8(qweight_int, bits).contiguous()
        scales_packed = scales.reshape(qparam_shape).contiguous()
        if symmetric:
            if not torch.all(zero_points == quantizer.midq):
                raise ValueError("Zero points must equal midq for symmetric quantization")
            qzeros_packed = None
        else:
            qzeros_packed = pack_to_uint8(zero_points.reshape(qparam_shape), bits).contiguous()

        return cls(
            qweight=qweight_packed,
            scales=scales_packed,
            qzeros=qzeros_packed,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            shape=tuple(weight.shape),
            dtype=scales_packed.dtype,
        )

    @classmethod
    def from_packed(
        cls,
        qweight: torch.Tensor,
        scales: torch.Tensor,
        qzeros: torch.Tensor | None,
        bits: int,
        group_size: int,
        symmetric: bool,
        shape: tuple[int, ...],
        dtype: torch.dtype | None = None,
        is_placeholder: bool = False,
    ) -> QuantTensor:
        """Reconstruct a ``QuantTensor`` from already-packed buffers."""
        return cls(
            qweight=qweight,
            scales=scales,
            qzeros=qzeros,
            bits=bits,
            group_size=group_size,
            symmetric=symmetric,
            shape=shape,
            dtype=dtype if dtype is not None else scales.dtype,
            is_placeholder=is_placeholder,
        )

    # ------------------------------------------------------------------
    # Dequantization
    # ------------------------------------------------------------------

    def to_dense(self) -> torch.Tensor:
        """Unpack + dequantize into a dense FP tensor of ``self.dtype``."""
        return _dequantize(self)

    # ------------------------------------------------------------------
    # Flatten / Unflatten for torch.compile and friends
    # ------------------------------------------------------------------

    def __tensor_flatten__(self):
        names = ["qweight", "scales"]
        if self.qzeros is not None:
            names.append("qzeros")
        meta = {
            "bits": self.bits,
            "group_size": self.group_size,
            "symmetric": self.symmetric,
            "shape": tuple(self.shape),
            "dtype": self.dtype,
            "has_qzeros": self.qzeros is not None,
            "is_placeholder": self.is_placeholder,
        }
        return names, meta

    @classmethod
    def __tensor_unflatten__(cls, inner_tensors, meta, outer_size, outer_stride):
        return cls(
            qweight=inner_tensors["qweight"],
            scales=inner_tensors["scales"],
            qzeros=inner_tensors["qzeros"] if meta["has_qzeros"] else None,
            bits=meta["bits"],
            group_size=meta["group_size"],
            symmetric=meta["symmetric"],
            shape=meta["shape"],
            dtype=meta["dtype"],
            is_placeholder=meta.get("is_placeholder", False),
        )

    # ------------------------------------------------------------------
    # _apply_fn_to_data — propagate per-tensor transforms (.to, detach…)
    # through every inner buffer
    # ------------------------------------------------------------------

    def _apply_fn_to_data(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> QuantTensor:
        new_qweight = fn(self.qweight)
        new_scales = fn(self.scales)
        new_qzeros = fn(self.qzeros) if self.qzeros is not None else None
        return QuantTensor(
            qweight=new_qweight,
            scales=new_scales,
            qzeros=new_qzeros,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            shape=tuple(self.shape),
            dtype=new_scales.dtype,
            is_placeholder=self.is_placeholder,
        )

    # ------------------------------------------------------------------
    # Dispatch
    # ------------------------------------------------------------------

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        handler = _TORCH_FN_TABLE.get(func)
        if handler is not None:
            return handler(*args, **kwargs)
        # Fall through to __torch_dispatch__ for everything else.
        return super().__torch_function__(func, types, args, kwargs)

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        aten = torch.ops.aten

        if func in (aten.detach.default, aten.clone.default, aten.alias.default, aten.contiguous.default):
            self_ = args[0]
            extra_args = args[1:]
            return self_._apply_fn_to_data(lambda x: func(x, *extra_args, **kwargs))

        if func is aten._to_copy.default:
            self_ = args[0]
            dtype = kwargs.get("dtype")
            device = kwargs.get("device")

            def _move(x: torch.Tensor) -> torch.Tensor:
                copy_kwargs: dict[str, Any] = {}
                if device is not None:
                    copy_kwargs["device"] = device
                # only scales are real-dtype; keep qweight/qzeros as uint8
                if dtype is not None and x.is_floating_point():
                    copy_kwargs["dtype"] = dtype
                return func(x, **copy_kwargs) if copy_kwargs else x

            return self_._apply_fn_to_data(_move)

        if func is aten.copy_.default:
            self_ = args[0]
            src = args[1]
            if not isinstance(src, QuantTensor):
                raise TypeError(f"Cannot copy_ a non-QuantTensor source into a QuantTensor (got {type(src)})")
            self_.qweight.copy_(src.qweight)
            self_.scales.copy_(src.scales)
            if self_.qzeros is not None and src.qzeros is not None:
                self_.qzeros.copy_(src.qzeros)
            # Mirror the source's placeholder state: copying real data into ``self_``
            # makes it real too, so a later in-place initializer must raise instead of
            # silently no-oping (the no-op is only valid while no real data has landed).
            # Conversely, copying *from* a placeholder leaves ``self_`` a placeholder
            # (its content is still throwaway dummy data).
            self_.is_placeholder = src.is_placeholder
            return self_

        if func in _NOOP_INIT_OPS and isinstance(args[0], QuantTensor):
            self_ = args[0]
            if self_.is_placeholder:
                # In-place random/constant weight initializers (e.g. ``nn.init.normal_``)
                # can reach here via HF's ``PreTrainedModel._initialize_weights`` -- it
                # unconditionally (re-)initializes every parameter that isn't yet in the
                # checkpoint's key set, which includes our *placeholder* QuantTensor param
                # before ``from_pretrained`` fills the real ``<pname>_qweight`` / ``_scales``
                # / ``_qzeros`` buffers from the checkpoint (see ``OliveHfQuantizer``). The
                # placeholder's numeric content is immediately overwritten by that buffer
                # load, so these initializers are safe (and required) no-ops here rather
                # than a real 3D op that would need to dequantize/reify the expert tensor.
                return self_
            # A real (non-placeholder) QuantTensor already holds real quantized data: an
            # in-place initializer here would silently discard it (the packed buffers
            # cannot represent an arbitrary in-place mutation), so raise instead of
            # no-oping.
            raise RuntimeError(
                f"In-place initializer {func} is not supported on a quantized QuantTensor "
                f"(shape={tuple(self_.shape)}, bits={self_.bits}). Quantized storage cannot "
                "represent an arbitrary in-place mutation; the call would be silently "
                "discarded. Re-quantize from a dense tensor instead "
                "(``QuantTensor.from_float(...)``), or mutate the packed buffers "
                "(``.qweight`` / ``.scales`` / ``.qzeros``) directly."
            )

        # Fallback: dequantize any QuantTensor args and re-dispatch.
        new_args = [_maybe_dense(a) for a in args]
        new_kwargs = {k: _maybe_dense(v) for k, v in kwargs.items()}
        return func(*new_args, **new_kwargs)

    # Friendlier repr — full dequant would defeat the purpose.
    def __repr__(self) -> str:  # pragma: no cover - trivial
        return (
            f"QuantTensor(shape={tuple(self.shape)}, dtype={self.dtype}, device={self.device}, "
            f"bits={self.bits}, group_size={self.group_size}, symmetric={self.symmetric})"
        )


_MOE_EAGER_OOM_MSG = (
    "Unsupported eager op on a 3D fused-MoE QuantTensor would require fully dequantizing "
    "the expert tensor (OOM risk) and is refused. Slice the expert dim first (e.g. "
    "`weight[expert_ids]`), or call `.to_dense()` explicitly outside a memory-sensitive path."
)


def _maybe_dense(x: Any) -> Any:
    if isinstance(x, QuantTensor):
        if x.dim() >= 3:
            # An unregistered op reaching this generic fallback would otherwise fully
            # dequantize the (potentially huge) fused-3D expert tensor. Refuse in both
            # eager mode (OOM risk) and under ONNX export (storage-only contract).
            raise RuntimeError(_MOE_ONNX_EXPORT_MSG if torch.onnx.is_in_onnx_export() else _MOE_EAGER_OOM_MSG)
        return x.to_dense()
    return x


# ----------------------------------------------------------------------
# Torch-function overrides
# ----------------------------------------------------------------------


@implements(F.linear)
def _linear(input: torch.Tensor, weight: QuantTensor, bias: torch.Tensor | None = None) -> torch.Tensor:  # noqa: A002
    if torch.onnx.is_in_onnx_export():
        raise RuntimeError(
            "Olive QuantTensor cannot be traced by torch.onnx.export directly. "
            "Use olive.common.hf.quant.make_export_compatible_quant(model, dynamo=...) "
            "before exporting, which replaces nn.Linear modules backed by a "
            "QuantTensor with an exportable QuantLinearNbit nn.Module."
        )
    if weight.dim() != 2:
        raise RuntimeError(
            "F.linear expects a 2D weight; got a "
            f"{weight.dim()}D QuantTensor. For 3D fused MoE experts, slice the leading "
            "dim first (e.g. `weight[expert_idx]`)."
        )
    dense = weight.to_dense().to(input.dtype)
    return F.linear(input, dense, bias)


@implements(F.embedding)
def _embedding(
    input: torch.Tensor,  # noqa: A002
    weight: QuantTensor,
    padding_idx: int | None = None,
    max_norm: float | None = None,
    norm_type: float = 2.0,
    scale_grad_by_freq: bool = False,
    sparse: bool = False,
) -> torch.Tensor:
    if torch.onnx.is_in_onnx_export():
        raise RuntimeError(
            "Olive QuantTensor cannot be traced by torch.onnx.export directly. "
            "Use olive.common.hf.quant.make_export_compatible_quant(model, dynamo=...) "
            "before exporting, which replaces nn.Embedding modules backed by a "
            "QuantTensor with an exportable QuantEmbeddingNbit nn.Module."
        )
    if weight.dim() != 2:
        raise RuntimeError(f"F.embedding expects a 2D weight; got a {weight.dim()}D QuantTensor.")
    dense = weight.to_dense()
    return F.embedding(input, dense, padding_idx, max_norm, norm_type, scale_grad_by_freq, sparse)


def _quant_metadata(t: QuantTensor) -> tuple:
    """Non-tensor identity of a ``QuantTensor`` (everything except the buffer contents)."""
    return (t.bits, t.group_size, t.symmetric, tuple(t.shape), t.dtype, t.qzeros is not None)


@implements(torch.equal)
def _equal(a, b) -> bool:
    """Structural equality that never materializes the dense weight.

    ``transformers>=5``'s ``PreTrainedModel.tie_weights`` calls ``torch.equal`` on the two
    tied word-embedding parameters *during* ``from_pretrained`` (in
    ``_finalize_model_loading``, i.e. **before** the quantizer's
    ``_process_model_after_weight_loading`` hook runs). At that point the placeholder
    ``QuantTensor``'s inner buffers may still live on the ``meta`` device, and the generic
    ``_maybe_dense`` fallback would dequantize them -- which hard-fails with
    ``NotImplementedError: aten::equal ... with Meta tensors``. Comparing the packed buffers
    directly (with an object-identity fast path for the tied case) avoids both the crash and
    a needless full dequantization of a potentially huge weight.
    """
    if a is b:
        return True
    if isinstance(a, QuantTensor) and isinstance(b, QuantTensor):
        if _quant_metadata(a) != _quant_metadata(b):
            return False
        pairs = [(a.qweight, b.qweight), (a.scales, b.scales)]
        if a.qzeros is not None:
            pairs.append((a.qzeros, b.qzeros))
        for x, y in pairs:
            if x is y:
                continue
            if x.device.type == "meta" or y.device.type == "meta":
                # ``meta`` tensors carry no data, so only storage identity is knowable and
                # two distinct meta buffers cannot be proven equal.
                return False
            if not torch.equal(x, y):
                return False
        return True
    if a.device.type == "meta" or b.device.type == "meta":
        # A mixed QuantTensor / dense comparison can also occur while Transformers is
        # loading an untied checkpoint. Meta tensors carry no values, so the parameters
        # cannot be proven equal and must remain untied.
        return False
    # Mixed QuantTensor / dense comparison: fall back to the dense path.
    return torch.equal(_maybe_dense(a), _maybe_dense(b))


@implements(torch.matmul, torch.Tensor.matmul)
def _matmul(a, b):
    if torch.onnx.is_in_onnx_export() and (isinstance(a, QuantTensor) or isinstance(b, QuantTensor)):
        raise RuntimeError(
            "ONNX export of matmul on a QuantTensor is not supported in Olive. "
            "Olive's MoE quantization is storage-only; use Mobius to emit "
            "com.microsoft.QMoE or a per-expert MatMulNBits loop."
        )
    return torch.matmul(_maybe_dense(a), _maybe_dense(b))


@implements(torch.bmm)
def _bmm(a, b):
    if torch.onnx.is_in_onnx_export() and (isinstance(a, QuantTensor) or isinstance(b, QuantTensor)):
        raise RuntimeError(
            "ONNX export of bmm on a QuantTensor is not supported in Olive. "
            "Olive's MoE quantization is storage-only; use Mobius to emit "
            "com.microsoft.QMoE or a per-expert MatMulNBits loop."
        )
    return torch.bmm(_maybe_dense(a), _maybe_dense(b))


@implements(torch.Tensor.__getitem__)
def _getitem(self: QuantTensor, idx):
    # Selecting along the leading (expert) dim of a 3D QuantTensor keeps the result on the
    # quantized fast path — for scalar ``weight[int]`` selection, a 0-D/1-D integer tensor
    # or list of ints (e.g. a flattened ``(tokens,)`` expert-id gather), a boolean/uint8
    # mask, a slice, and tuple-form leading-only indexing ``weight[expert_ids, :, :]``.
    # Indexing the packed buffers directly avoids dequantizing the *entire* expert tensor
    # (the OOM risk that would otherwise defeat MoE quantization).
    #
    # Rank >= 2 integer-tensor indices (e.g. an un-flattened ``(tokens, k)`` top-k routing
    # tensor) are deliberately NOT accepted here, even though the packed-buffer arithmetic
    # below would happily produce a >3D QuantTensor "view" for them (see #2598 item 4):
    # every dense-consuming op (``_dequantize``, ``F.linear``/``F.embedding``, and the
    # generic ``_maybe_dense`` OOM guard used by ``matmul``/``bmm``/every unregistered op)
    # refuses any QuantTensor with rank > 3, so such a result would be a dead end — it
    # could never be dequantized or fed into any op. There is currently no caller of this
    # rank in the repo, and no validated design for how the OOM guard should distinguish a
    # small already-gathered batch (safe to dequantize) from the full multi-GB expert
    # tensor (unsafe) once rank stops being a reliable proxy for size. Callers needing
    # multi-dim batched expert selection should flatten to 1-D first (e.g.
    # ``weight[expert_ids.flatten()]``), consume the result, then reshape the *dense
    # output* back to the original batch shape — the standard "flatten batch dims / do the
    # op / unflatten" pattern. Revisit this once a real consumer (e.g. a vectorized
    # GPTQ-MoE forward/calibration path) exists to validate the OOM-guard redesign against.
    if self.dim() == 3 and _indexes_leading_dim_only(idx, int(self.shape[0])):
        new_qweight = self.qweight[idx]
        new_scales = self.scales[idx]
        new_qzeros = self.qzeros[idx] if self.qzeros is not None else None
        # Leading batch dim of the selection (``()`` for int, ``(K,)`` for a 0-D/1-D
        # tensor/list/mask index) — always rank <= 1 now that ``_indexes_leading_dim_only``
        # rejects rank >= 2 integer tensors (#2598 item 4).
        leading = tuple(new_qweight.shape[:-2])
        new_shape = (*leading, *tuple(self.shape[1:]))
        # Belt-and-braces: catches any residual arithmetic slip in ``_indexes_leading_dim_only``
        # (or a future extension of it) before it produces a QuantTensor whose ``.shape``
        # metadata disagrees with the actual packed-buffer ranks.
        if tuple(new_scales.shape[:-2]) != leading:
            raise RuntimeError(
                f"Internal error: QuantTensor index {idx!r} produced inconsistent buffer ranks "
                f"(qweight {tuple(new_qweight.shape)} vs scales {tuple(new_scales.shape)})."
            )
        return QuantTensor(
            qweight=new_qweight,
            scales=new_scales,
            qzeros=new_qzeros,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            shape=new_shape,
            dtype=self.dtype,
            is_placeholder=self.is_placeholder,
        )
    if self.dim() >= 3:
        # Any other 3D indexing pattern would require fully dequantizing the expert tensor
        # (this also covers rank >= 2 integer-tensor indices, e.g. an un-flattened
        # ``(tokens, k)`` top-k routing tensor -- see #2598 item 4 / the note above).
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(_MOE_ONNX_EXPORT_MSG)
        raise RuntimeError(
            f"Unsupported indexing pattern {idx!r} on a {self.dim()}D fused-MoE QuantTensor. "
            "Only leading-dim (expert) selection preserves quantized storage (boolean masks "
            "must be 1-D and match the expert dim; integer tensor/list indices must be 0-D "
            "or 1-D -- flatten a multi-dim batch of expert ids first, e.g. "
            "`weight[expert_ids.flatten()]`, and reshape the *dense output* back afterward); "
            "other indexing would require fully dequantizing the expert tensor and is "
            "refused to avoid silent memory blow-up. Slice the expert dim first (e.g. "
            "`weight[expert_ids]`)."
        )
    return self.to_dense()[idx]


def _is_bool_index(idx: Any) -> bool:
    """Whether ``idx`` is (or contains, for a Python list) a boolean mask.

    ``torch.uint8`` tensors are included here on purpose: PyTorch's indexing
    semantics treat a ``uint8`` tensor index as a *legacy boolean mask*
    (``nonzero()``-style selection, with a deprecation warning), not as an
    integer gather index — unlike every other unsigned/narrow integer dtype
    (``int8``/``int16``/``uint16``/``uint32``/``uint64``), which torch simply
    refuses to index with. Without this, a 1-D uint8 index of the wrong
    length, or a 0-D uint8 scalar (e.g. ``expert_idx[0]`` in real MoE routing
    code), would fall through to the generic "any non-float/complex tensor is
    a safe gather index" branch below and silently produce a shape-inserting
    result instead of raising or correctly selecting.
    """
    if isinstance(idx, torch.Tensor):
        return idx.dtype in (torch.bool, torch.uint8)
    if isinstance(idx, list):
        return any(isinstance(e, bool) for e in idx)
    return isinstance(idx, bool)


def _indexes_leading_dim_only(idx: Any, leading_dim: int) -> bool:
    """Whether ``idx`` selects along a single (leading) dimension only.

    Safe forms — each consumes *exactly one* dimension, so ``_getitem``'s
    ``(*index_batch_dims, *self.shape[1:])`` shape arithmetic stays valid:

    * ``int`` / ``slice`` / list of ``int`` / a **0-D or 1-D integer** tensor *except*
      ``uint8`` (e.g. a flattened ``(tokens,)`` expert-id gather);
    * a **1-D boolean (or legacy ``uint8``) mask** whose length equals ``leading_dim``.

    Rejected: rank >= 2 integer tensors (see the module-level note in ``_getitem`` on why
    un-flattened multi-dim batch indices like a ``(tokens, k)`` top-k routing tensor are
    deliberately not supported here — #2598 item 4), rank != 1 boolean/``uint8`` masks and
    length-mismatched masks (they consume more or fewer than one dim — the source of the
    metadata/data shape mismatch), ``bool`` scalars (which *add* a dim), and lists
    containing ``bool`` (torch treats them as masks). ``uint8`` tensors are treated as
    masks (never as gather indices) because that is PyTorch's own legacy indexing
    semantics for that dtype — see ``_is_bool_index``.

    Tuples are accepted only when the head is a valid leading index and every remaining
    element is a full slice (``:``).
    """
    if isinstance(idx, tuple):
        if not idx or len(idx) > 3:
            return False
        head, *rest = idx
        if not _indexes_leading_dim_only(head, leading_dim):
            return False
        return all(isinstance(r, slice) and r == slice(None) for r in rest)
    if _is_bool_index(idx):
        if isinstance(idx, torch.Tensor):
            return idx.dim() == 1 and idx.shape[0] == leading_dim
        # Python list-of-bools mask.
        return isinstance(idx, list) and all(isinstance(e, bool) for e in idx) and len(idx) == leading_dim
    if isinstance(idx, slice):
        return True
    if isinstance(idx, int):  # ``bool`` already handled above
        return True
    if isinstance(idx, list):
        return all(isinstance(e, int) and not isinstance(e, bool) for e in idx)
    # Rank >= 2 integer tensors are rejected here on purpose -- see the docstring above and
    # the note in ``_getitem`` (#2598 item 4): a rank-2+ leading index would produce a
    # QuantTensor with rank > 3, which every dense-consuming op in this module refuses.
    return isinstance(idx, torch.Tensor) and not idx.is_floating_point() and not idx.is_complex() and idx.dim() <= 1


@implements(torch.Tensor.to)
def _to(self: QuantTensor, *args, **kwargs):
    # Use torch's own _parse_to to robustly resolve the (device, dtype,
    # non_blocking, convert_to_format) tuple — covers every signature
    # including nn.Module.to's ``t.to(None, dtype, non_blocking)``.
    device, dtype, _, _ = torch._C._nn._parse_to(*args, **kwargs)  # type: ignore[attr-defined]

    if device is None and dtype is None:
        return self

    def _move(x: torch.Tensor) -> torch.Tensor:
        move_kwargs: dict[str, Any] = {}
        if device is not None:
            move_kwargs["device"] = device
        if dtype is not None and x.is_floating_point():
            move_kwargs["dtype"] = dtype
        return x.to(**move_kwargs) if move_kwargs else x

    return self._apply_fn_to_data(_move)


# ----------------------------------------------------------------------
# Movement / view ops
# ----------------------------------------------------------------------
# Storage-only quantization intercepts ``F.linear`` / ``F.embedding`` (and integer
# expert selection). Shape-movement ops (transpose / reshape / view / permute / …) are
# *not* implemented against the packed layout: letting them fall through to the default
# tensor-subclass machinery silently produces a malformed ``QuantTensor`` (constructed via
# ``__new__`` without the packed metadata), which then fails deep inside an unrelated op.
# We reject them centrally with a clear, actionable error instead — and under ONNX export
# surface the same MoE-export guidance as every other unsupported 3D op.
_UNSUPPORTED_MOVEMENT_FNS = (
    torch.transpose,
    torch.Tensor.transpose,
    torch.t,
    torch.Tensor.t,
    torch.permute,
    torch.Tensor.permute,
    torch.swapaxes,
    torch.Tensor.swapaxes,
    torch.movedim,
    torch.Tensor.movedim,
    torch.reshape,
    torch.Tensor.reshape,
    torch.Tensor.view,
    torch.flatten,
    torch.Tensor.flatten,
    torch.Tensor.expand,
    torch.squeeze,
    torch.Tensor.squeeze,
    torch.unsqueeze,
    torch.Tensor.unsqueeze,
)


@implements(*_UNSUPPORTED_MOVEMENT_FNS)
def _unsupported_movement(*args, **kwargs):
    self = next((a for a in args if isinstance(a, QuantTensor)), None)
    if self is not None and self.dim() >= 3 and torch.onnx.is_in_onnx_export():
        raise RuntimeError(_MOE_ONNX_EXPORT_MSG)
    raise RuntimeError(
        "Shape-movement / view ops (transpose, reshape, view, permute, flatten, expand, …) are "
        "not supported on an Olive QuantTensor: quantization is storage-only and these ops would "
        "require fully dequantizing the packed weight. Access the dense weight explicitly via "
        "``.to_dense()`` if you really need to reshape it (e.g. outside a memory-sensitive path)."
    )
