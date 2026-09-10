# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Convert ``com.microsoft::MoE`` nodes to ``com.microsoft::QMoE``.

The ``MoE`` op carries the per-expert ``fc1_experts_weights`` and
``fc2_experts_weights`` as 3-D fp16 / bf16 / fp32 initializers. The
``QMoE`` op accepts the same logical inputs but with the weights packed
as symmetric int4 (or int8) plus per-row or block-wise scale tensors,
laid out in the CUTLASS ``fpA_intB`` mixed-precision GEMM format that
the CUDA / CPU QMoE kernels consume.

This pass:

1. Walks the graph and finds every ``com.microsoft::MoE`` node whose
   ``fc1_experts_weights`` and ``fc2_experts_weights`` are static 3-D
   initializers.
2. For each expert, symmetrically quantizes the per-expert weight slice
   using ORT's ``quantize_matmul_4bits`` / ``quantize_matmul_8bits``
   pybind helper.
3. Stacks the per-expert quantized weights and scales into 3-D /
   2-D / 3-D initializers (matching the QMoE schema) and registers them
   on the graph.
4. Replaces the ``MoE`` node with a ``QMoE`` node carrying the original
   activation / routing attributes plus ``expert_weight_bits`` and
   ``block_size``.

The resulting model targets ORT ≥ 1.28 / nightly post #28467; the QMoE
kernel is currently CUDA-only (plus an experimental CPU fallback).
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import onnx_ir as ir
from onnx_ir.passes.common.unused_removal import RemoveUnusedNodesPass

from olive.constants import MSFT_DOMAIN
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

if TYPE_CHECKING:
    from olive.hardware.accelerator import AcceleratorSpec
    from olive.model import ONNXModelHandler

logger = logging.getLogger(__name__)


_MOE_OP_TYPE = "MoE"
_QMOE_OP_TYPE = "QMoE"

# Input slot layout for both ``com.microsoft::MoE`` and ``com.microsoft::QMoE``
# (the QMoE op interleaves scale tensors after each weight tensor):
#
#   MoE:  [input, router_probs, fc1_W, fc1_b, fc2_W, fc2_b, fc3_W, fc3_b]
#   QMoE: [input, router_probs,
#          fc1_W, fc1_scales, fc1_zp, fc1_b,
#          fc2_W, fc2_scales, fc2_zp, fc2_b,
#          fc3_W, fc3_scales, fc3_zp, fc3_b]  (zp optional)
_MOE_INPUT_INDEX = {
    "input": 0,
    "router_probs": 1,
    "fc1_W": 2,
    "fc1_b": 3,
    "fc2_W": 4,
    "fc2_b": 5,
    "fc3_W": 6,
    "fc3_b": 7,
}


class OnnxMoEQuantization(Pass):
    """Convert ``com.microsoft::MoE`` ops to ``com.microsoft::QMoE``.

    Quantizes the per-expert FC1 / FC2 weight initializers to symmetric
    int4 (default) or int8 and rewires each ``MoE`` node to a ``QMoE``
    node, preserving all routing / activation attributes.
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description=("Number of bits per quantized weight. Supported: 4 (default) or 8."),
            ),
            "block_size": PassConfigParam(
                type_=int,
                default_value=0,
                description=(
                    "Block size along the K dimension. 0 means per-row scales (one "
                    "scale per output channel). When > 0, must be a power of two "
                    "≥ 16 and the K dimension of each expert weight must be "
                    "divisible by it."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list[str] | None,
                default_value=None,
                description="List of MoE node names to leave unquantized.",
            ),
            "force_arch": PassConfigParam(
                type_=int,
                default_value=80,
                description=(
                    "Target CUDA SM version for the CUTLASS weight prepacking "
                    "(80 = Ampere, 90 = Hopper). Most deployments are forward "
                    "compatible at sm_80."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self,
        model: ONNXModelHandler,
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, "model.onnx")

        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1

        if config.bits not in (4, 8):
            raise ValueError(f"OnnxMoEQuantization: bits must be 4 or 8, got {config.bits}.")
        if config.block_size < 0:
            raise ValueError(f"OnnxMoEQuantization: block_size must be ≥ 0, got {config.block_size}.")
        if config.block_size > 0 and (config.block_size < 16 or config.block_size & (config.block_size - 1)):
            raise ValueError(
                f"OnnxMoEQuantization: block_size must be 0 or a power of two ≥ 16, got {config.block_size}."
            )

        converted = _convert_moe_to_qmoe(
            ir_model,
            bits=config.bits,
            block_size=config.block_size,
            nodes_to_exclude=config.nodes_to_exclude or [],
            force_arch=config.force_arch,
        )
        logger.info("OnnxMoEQuantization: converted %d MoE node(s) to QMoE.", converted)

        # Drop the original 3-D fp16 weight initializers (now replaced by
        # quantized uint8 weight + fp16 scale initializers). Defer to
        # onnx_ir's standard dead-code elimination pass so this stays
        # consistent with other consumers of the IR.
        RemoveUnusedNodesPass()(ir_model)

        return ir_model_to_olive_model(ir_model, output_model_path, config)


# ---------------------------------------------------------------------------
# Graph rewrite helpers (module-private, per Google-style guide preference for
# free functions over static / class methods when no shared state is involved)
# ---------------------------------------------------------------------------


def _convert_moe_to_qmoe(
    ir_model: ir.Model,
    bits: int,
    block_size: int,
    nodes_to_exclude: list[str],
    force_arch: int,
) -> int:
    """Walk ``ir_model.graph`` and rewrite every MoE node to a QMoE node.

    Returns the number of nodes successfully converted. Nodes whose weights
    can't be statically quantized (e.g. dynamic weight inputs, shape that
    doesn't divide cleanly into pack tiles) are skipped with a logger
    warning rather than aborting the whole pass.
    """
    graph = ir_model.graph
    initializers: dict[str, ir.Value] = dict(graph.initializers)
    excluded = set(nodes_to_exclude)
    converted = 0

    for node in list(graph.all_nodes()):
        if node.op_type != _MOE_OP_TYPE or node.domain != MSFT_DOMAIN:
            continue
        if node.name in excluded:
            logger.debug("Skipping MoE node %s (in nodes_to_exclude).", node.name)
            continue

        try:
            qmoe_node = _convert_single_moe(node, initializers, bits=bits, block_size=block_size, force_arch=force_arch)
        except _UnsupportedMoEError as exc:
            logger.warning("Skipping MoE node %s: %s", node.name or "<anon>", exc)
            continue

        ir.convenience.replace_nodes_and_values(graph, node, [node], [qmoe_node], node.outputs, qmoe_node.outputs)
        converted += 1

    return converted


def _convert_single_moe(
    node: ir.Node,
    initializers: dict[str, ir.Value],
    bits: int,
    block_size: int,
    force_arch: int,
) -> ir.Node:
    """Build the QMoE replacement for a single MoE node.

    Quantizes the per-expert FC1/FC2 initializers and registers the new
    initializers on the same graph as ``node``. Raises
    ``_UnsupportedMoEError`` if the node's shape, dtype, or input set
    isn't something this pass can handle (the caller logs and skips).
    """
    fc1_w_value = _get_input(node, _MOE_INPUT_INDEX["fc1_W"])
    fc2_w_value = _get_input(node, _MOE_INPUT_INDEX["fc2_W"])
    fc1_array = _require_initializer(fc1_w_value, "fc1_experts_weights", initializers)
    fc2_array = _require_initializer(fc2_w_value, "fc2_experts_weights", initializers)

    if fc1_array.ndim != 3 or fc2_array.ndim != 3:
        raise _UnsupportedMoEError(f"Expected 3-D weights; got fc1.ndim={fc1_array.ndim}, fc2.ndim={fc2_array.ndim}.")

    num_experts = fc1_array.shape[0]
    if fc2_array.shape[0] != num_experts:
        raise _UnsupportedMoEError(f"fc1/fc2 num_experts disagree: {fc1_array.shape[0]} vs {fc2_array.shape[0]}.")

    fc1_qweights, fc1_scales = _quantize_stacked_weights(
        fc1_array, bits=bits, block_size=block_size, force_arch=force_arch
    )
    fc2_qweights, fc2_scales = _quantize_stacked_weights(
        fc2_array, bits=bits, block_size=block_size, force_arch=force_arch
    )

    graph = node.graph
    fc1_w_init = _make_initializer(f"{fc1_w_value.name}_q", fc1_qweights)
    fc1_s_init = _make_initializer(f"{fc1_w_value.name}_scales", fc1_scales)
    fc2_w_init = _make_initializer(f"{fc2_w_value.name}_q", fc2_qweights)
    fc2_s_init = _make_initializer(f"{fc2_w_value.name}_scales", fc2_scales)
    for init in (fc1_w_init, fc1_s_init, fc2_w_init, fc2_s_init):
        graph.register_initializer(init)

    fc1_bias = _maybe_input(node, _MOE_INPUT_INDEX["fc1_b"])
    fc2_bias = _maybe_input(node, _MOE_INPUT_INDEX["fc2_b"])
    fc3_w = _maybe_input(node, _MOE_INPUT_INDEX["fc3_W"])
    if fc3_w is not None:
        raise _UnsupportedMoEError("fc3 inputs are not yet supported by this pass.")

    # QMoE input layout (zero_points stay absent because we use symmetric
    # int4/int8):
    #   0: input
    #   1: router_probs
    #   2: fc1_experts_weights    (quantized)
    #   3: fc1_scales
    #   4: fc1_zero_points        (None — symmetric)
    #   5: fc1_experts_bias
    #   6: fc2_experts_weights    (quantized)
    #   7: fc2_scales
    #   8: fc2_zero_points        (None)
    #   9: fc2_experts_bias
    qmoe_inputs = [
        _get_input(node, _MOE_INPUT_INDEX["input"]),
        _get_input(node, _MOE_INPUT_INDEX["router_probs"]),
        fc1_w_init,
        fc1_s_init,
        None,
        fc1_bias,
        fc2_w_init,
        fc2_s_init,
        None,
        fc2_bias,
    ]

    new_attrs = list(node.attributes.values())
    new_attrs.append(ir.AttrInt64("expert_weight_bits", bits))
    if block_size > 0:
        new_attrs.append(ir.AttrInt64("block_size", block_size))
    # ``quant_type`` defaults to ``"int"`` in the schema; emit it
    # explicitly so future schema revisions changing the default don't
    # silently alter behaviour for our exported models.
    if not any(a.name == "quant_type" for a in node.attributes.values()):
        new_attrs.append(ir.AttrString("quant_type", "int"))

    output_value = ir.Value(name=node.outputs[0].name)
    return ir.Node(
        domain=MSFT_DOMAIN,
        op_type=_QMOE_OP_TYPE,
        inputs=qmoe_inputs,
        attributes=new_attrs,
        outputs=[output_value],
        name=node.name + "_QMoE" if node.name else None,
    )


# ---------------------------------------------------------------------------
# Small graph / value helpers
# ---------------------------------------------------------------------------


class _UnsupportedMoEError(Exception):
    """Raised when a particular MoE node can't be converted by this pass."""


def _get_input(node: ir.Node, idx: int) -> ir.Value:
    if idx >= len(node.inputs) or node.inputs[idx] is None:
        raise _UnsupportedMoEError(f"Missing required input at slot {idx}.")
    return node.inputs[idx]


def _maybe_input(node: ir.Node, idx: int) -> ir.Value | None:
    """Return the optional input at ``idx``, treating empty / missing slots as absent.

    ONNX represents an unset optional input either as a slot past the end of the
    inputs list, or as an in-place placeholder with an empty name. ``onnx_ir``
    typically maps the latter to ``None``, but defensively handle the
    ``ir.Value(name="")`` case too so callers can rely on ``None`` meaning absent.
    """
    if idx >= len(node.inputs):
        return None
    value = node.inputs[idx]
    if value is None or not value.name:
        return None
    return value


def _require_initializer(value: ir.Value, what: str, initializers: dict[str, ir.Value]) -> np.ndarray:
    init = initializers.get(value.name)
    if init is None or init.const_value is None:
        raise _UnsupportedMoEError(f"{what} ({value.name!r}) is not a static initializer.")
    return init.const_value.numpy()


def _make_initializer(name: str, array: np.ndarray) -> ir.Value:
    tensor = ir.Tensor(array, name=name)
    return ir.Value(
        name=name,
        type=ir.TensorType(tensor.dtype),
        shape=ir.Shape(array.shape),
        const_value=tensor,
    )


def _quantize_stacked_weights(
    weights_3d: np.ndarray, bits: int, block_size: int, force_arch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize each expert's [N, K] slice and stack along axis 0.

    Returns a tuple ``(packed_weights, scales)`` where:

    - ``packed_weights`` has shape ``[E, K, N // pack_size]`` (uint8),
      laid out in the CUTLASS ``fpA_intB`` mixed-precision GEMM format
      expected by the QMoE kernels.
    - ``scales`` has shape ``[E, N]`` for per-row scales, or
      ``[E, N, K // block_size]`` for block-wise scales (fp16).
    """
    if bits not in (4, 8):
        raise ValueError(f"bits must be 4 or 8, got {bits}")
    num_experts = weights_3d.shape[0]

    packed_per_expert = []
    scales_per_expert = []

    pack_fn = _load_cuda_pack_fn()

    for e in range(num_experts):
        weight = weights_3d[e]  # [N, K]
        packed, scale = _quantize_one_expert(
            weight, bits=bits, block_size=block_size, pack_fn=pack_fn, force_arch=force_arch
        )
        packed_per_expert.append(packed)
        scales_per_expert.append(scale)

    return np.stack(packed_per_expert, axis=0), np.stack(scales_per_expert, axis=0)


def _quantize_one_expert(
    weight: np.ndarray, bits: int, block_size: int, pack_fn, force_arch: int
) -> tuple[np.ndarray, np.ndarray]:
    """Quantize a single expert's ``[N, K]`` weight matrix.

    Mirrors the test harness in
    ``onnxruntime/test/python/transformers/test_qmoe_cuda.py:quant_dequant_blockwise``:

    1. Transpose to ``[K, N]`` (CUTLASS column-major convention).
    2. Call ORT's ``quantize_matmul_{bits}bits`` to produce per-block
       ``q_weight`` and ``scale``.
    3. Call ORT's ``pack_weights_for_cuda_mixed_gemm`` to permute /
       interleave the bytes into the fpA_intB layout the kernel reads.
    """
    from onnxruntime.capi import _pybind_state as _p

    quant_fn_name = f"quantize_matmul_{bits}bits"
    quantize = getattr(_p, quant_fn_name, None)
    if quantize is None:
        raise RuntimeError(
            f"onnxruntime.capi._pybind_state.{quant_fn_name} is not available; "
            "this Olive pass needs a recent build of onnxruntime."
        )

    # Promote fp16/bf16 to fp32 for the quantizer (bindings only accept
    # fp16 or fp32; bf16 isn't supported as a python numpy dtype).
    if weight.dtype in (np.float16, np.float32):
        weight_for_quant = weight
    else:
        weight_for_quant = weight.astype(np.float32)

    n, k = weight_for_quant.shape  # per-expert weight is [N, K]
    weight_t = np.ascontiguousarray(weight_for_quant.T)  # [K, N]

    effective_block = block_size if block_size > 0 else k
    if k % effective_block != 0:
        raise _UnsupportedMoEError(f"K ({k}) is not divisible by block_size ({effective_block}).")
    block_per_k = k // effective_block

    pack_factor = 8 // bits  # 2 for int4, 1 for int8
    if n % pack_factor != 0:
        raise _UnsupportedMoEError(f"N ({n}) must be divisible by pack_factor ({pack_factor}) for {bits}-bit packing.")
    if effective_block % pack_factor != 0:
        raise _UnsupportedMoEError(
            f"block_size ({effective_block}) must be divisible by pack_factor ({pack_factor}) for {bits}-bit packing."
        )
    blob_size = effective_block // pack_factor
    # The pybind quantize_matmul_{4,8}bits binding takes raw contiguous buffers
    # for ``scale`` and ``zero_point``; pybind11's buffer-protocol overload
    # accepts any shape with the same total element count. Matching the layout
    # used in the upstream ORT test harness
    # (onnxruntime/test/python/transformers/test_qmoe_cuda.py::quant_dequant_blockwise)
    # keeps the on-disk byte order obvious — 2-D ``[N, block_per_k]`` for scale,
    # 2-D ``[N, ceil(block_per_k / pack_factor)]`` for zero_point.
    q_weight = np.zeros((n, block_per_k, blob_size), dtype=np.uint8)
    scale = np.zeros((n, block_per_k), dtype=np.float32)
    zero_point = np.zeros((n, (block_per_k + pack_factor - 1) // pack_factor), dtype=np.uint8)

    # Symmetric quantization (kernel uses (q - bias) * scale internally).
    quantize(q_weight, weight_t, scale, zero_point, effective_block, n, k, True)  # pylint: disable=not-callable
    scale = np.abs(scale)

    # CUTLASS mixed-precision GEMM expects a specific byte layout.
    q_weight_flat = q_weight.reshape(n, -1)
    packed = pack_fn(q_weight_flat, n, k, bits, force_arch)
    packed = np.ascontiguousarray(packed.reshape(k, n // pack_factor)).view(np.uint8)

    # Squeeze trivial block dim to match the spec when block_size == 0:
    #   row-wise scales → [N]
    #   block-wise scales → [N, block_per_k]
    if block_size == 0:
        scale_out = scale.reshape(n).astype(np.float16)
    else:
        scale_out = scale.reshape(n, block_per_k).astype(np.float16)

    return packed, scale_out


def _load_cuda_pack_fn():
    """Locate ``pack_weights_for_cuda_mixed_gemm`` in the installed ORT.

    Raises a descriptive error if the binding isn't available; without
    it the produced model can't be loaded by the CUDA or CPU QMoE
    kernels because they read the CUTLASS pre-packed layout.
    """
    from onnxruntime.capi import _pybind_state as _p

    pack_fn = getattr(_p, "pack_weights_for_cuda_mixed_gemm", None)
    if pack_fn is None:
        raise RuntimeError(
            "OnnxMoEQuantization requires the CUTLASS weight-packing helper "
            "`pack_weights_for_cuda_mixed_gemm`, which is only exported by "
            "ONNX Runtime when built with CUDA support. Install onnxruntime-gpu "
            ">= 1.28 (or a nightly built from main with USE_CUDA after PR "
            "microsoft/onnxruntime#28467)."
        )
    return pack_fn
