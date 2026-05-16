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
  materialised in memory; only the packed buffers are allocated.
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


def implements(*torch_fns: Callable) -> Callable[[Callable], Callable]:
    """Register a torch-function override for ``QuantTensor``."""

    def decorator(fn: Callable) -> Callable:
        for torch_fn in torch_fns:
            _TORCH_FN_TABLE[torch_fn] = fn
        return fn

    return decorator


def _midq(bits: int) -> int:
    return 1 << (bits - 1)


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
    ) -> None:
        self.qweight = qweight
        self.scales = scales
        self.qzeros = qzeros
        self.bits = int(bits)
        self.group_size = int(group_size)
        self.symmetric = bool(symmetric)

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
            return self_

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


def _maybe_dense(x: Any) -> Any:
    if isinstance(x, QuantTensor):
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
    # Slicing the leading dim of a 3D QuantTensor returns a 2D
    # QuantTensor when possible — keeps MoE per-expert forwards
    # (`weight[expert_idx]`) on the quantized fast path.
    if self.dim() == 3 and isinstance(idx, int):
        new_shape = tuple(self.shape[1:])
        return QuantTensor(
            qweight=self.qweight[idx],
            scales=self.scales[idx],
            qzeros=self.qzeros[idx] if self.qzeros is not None else None,
            bits=self.bits,
            group_size=self.group_size,
            symmetric=self.symmetric,
            shape=new_shape,
            dtype=self.dtype,
        )
    return self.to_dense()[idx]


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
