# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# K-quant algorithm adapted from:
# https://github.com/microsoft/onnxruntime/blob/main/onnxruntime/python/tools/quantization/neural_compressor/weight_only.py
# Original reference: https://github.com/ggml-org/llama.cpp/blob/64eda5deb9859e87a020e56bab5d2f9ca956f1de/ggml/src/ggml-quants.c
# --------------------------------------------------------------------------
"""K-quant weight-only quantization pass.

Implements the k-quant algorithm natively in Olive using numpy, without
depending on onnxruntime's quantization modules.
"""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import onnx_ir as ir

from olive.constants import MSFT_DOMAIN, AccuracyLevel, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    ir_model_to_olive_model,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _kquant_quantize(data: np.ndarray, num_bits: int = 4, group_size: int = 32) -> tuple:
    """Quantize tensor per group using the k-quant algorithm.

    Tries GPU acceleration via cupy if available, otherwise falls back to CPU numpy.

    Args:
        data: input weight, will be reshaped to (-1, group_size).
        num_bits: quantization bit-width (4 or 8).
        group_size: number of elements per quantization group.

    Returns:
        q_weight: quantized weight, shape (nb, group_size).
        scale: per-group scale, shape (nb, 1).
        zero_point: per-group zero point as uint8, shape (nb, 1).

    """
    try:
        import cupy as cp
        import torch

        if torch.cuda.is_available():
            return _kquant_quantize_cuda(data, num_bits, group_size, cp)

        logger.warning("cupy is installed but CUDA is not available. Falling back to CPU k-quant quantization.")
    except ImportError:
        logger.info(
            "cupy/torch not found; using CPU k-quant quantization. "
            "Install cupy (https://cupy.dev/) and torch to accelerate with CUDA."
        )

    return _kquant_quantize_cpu(data, num_bits, group_size)


def _kquant_quantize_cpu(data: np.ndarray, num_bits: int = 4, group_size: int = 32) -> tuple:
    """CPU (numpy) implementation of k-quant quantization.

    Ref: https://github.com/ggml-org/llama.cpp/blob/64eda5deb9859e87a020e56bab5d2f9ca956f1de/ggml/src/ggml-quants.c

    """
    data = np.reshape(data, (-1, group_size)).astype(np.float32)
    maxq = 2**num_bits - 1
    minq = 0

    sum_x2 = np.sum(data**2, axis=1, keepdims=True)
    av_x = np.sqrt(sum_x2 / group_size)
    weights = np.add(av_x, np.abs(data))

    rmin = np.min(data, axis=1, keepdims=True)
    rmax = np.max(data, axis=1, keepdims=True)
    sum_w = np.sum(weights, axis=1, keepdims=True)
    sum_x = np.sum(weights * data, axis=1, keepdims=True)

    iscale = np.ones(rmax.shape, dtype=data.dtype)
    mask = rmin != rmax
    iscale[mask] = (maxq - minq) / (rmax[mask] - rmin[mask])
    scale = 1 / iscale
    quant_data = np.clip(np.round(iscale * (data - rmin)), minq, maxq)
    diff = scale * quant_data + rmin - data
    best_mad = np.sum(weights * diff**2, axis=1, keepdims=True)

    nstep = 20
    rdelta = 0.1
    rrmin = -1
    for is_ in range(nstep):
        iscale_new = np.ones(rmax.shape, dtype=data.dtype)
        factor = np.array([rrmin + rdelta * is_ + maxq - minq]).astype(data.dtype)[0]
        mask = rmin != rmax
        iscale_new[mask] = factor / (rmax[mask] - rmin[mask])
        quant_data_new = np.clip(np.round(iscale_new * (data - rmin)), minq, maxq)
        mul_weights_quant_data_new = weights * quant_data_new
        sum_l = np.sum(mul_weights_quant_data_new, axis=1, keepdims=True)
        sum_l2 = np.sum(mul_weights_quant_data_new * quant_data_new, axis=1, keepdims=True)
        sum_xl = np.sum(mul_weights_quant_data_new * data, axis=1, keepdims=True)
        D = np.subtract(sum_w * sum_l2, sum_l**2)  # noqa: N806

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D

        diff = this_scale * quant_data_new + this_min - data
        mad = np.sum(weights * diff**2, axis=1, keepdims=True)

        mad_1 = np.array(mad)
        best_mad_1 = np.array(best_mad)
        idx_to_replace = np.where(mad_1 < best_mad_1)[0]
        quant_data[idx_to_replace, :] = quant_data_new[idx_to_replace, :]
        best_mad[idx_to_replace] = mad[idx_to_replace]
        scale[idx_to_replace] = this_scale[idx_to_replace]
        rmin[idx_to_replace] = this_min[idx_to_replace]

    zero_point = np.clip(((-rmin) / scale).round(), 0, maxq).astype("uint8")
    scale = scale.astype(np.float64)
    q_weight = np.empty_like(data, dtype=scale.dtype)
    np.divide(data, scale, out=q_weight)
    np.add(q_weight, zero_point, out=q_weight)
    np.round(q_weight, out=q_weight)
    np.clip(q_weight, minq, maxq, out=q_weight)

    return q_weight, scale, zero_point


def _kquant_quantize_cuda(data: np.ndarray, num_bits: int, group_size: int, cp) -> tuple:
    """GPU (cupy) implementation of k-quant quantization.

    Same algorithm as the CPU version but runs on CUDA for faster processing.
    Results are transferred back to numpy arrays before returning.

    """
    data = cp.asarray(data)
    data = data.reshape((-1, group_size)).astype(cp.float32)
    maxq = 2**num_bits - 1
    minq = 0

    sum_x2 = cp.sum(data**2, axis=1, keepdims=True)
    av_x = cp.sqrt(sum_x2 / group_size)
    weights = cp.add(av_x, cp.abs(data))

    rmin = cp.min(data, axis=1, keepdims=True)
    rmax = cp.max(data, axis=1, keepdims=True)
    sum_w = cp.sum(weights, axis=1, keepdims=True)
    sum_x = cp.sum(weights * data, axis=1, keepdims=True)

    iscale = cp.ones(rmax.shape, dtype=data.dtype)
    mask = rmin != rmax
    iscale[mask] = (maxq - minq) / (rmax[mask] - rmin[mask])
    scale = 1 / iscale
    quant_data = cp.clip(cp.round(iscale * (data - rmin)), minq, maxq)
    diff = scale * quant_data + rmin - data
    best_mad = cp.sum(weights * diff**2, axis=1, keepdims=True)

    nstep = 20
    rdelta = 0.1
    rrmin = -1
    for is_ in range(nstep):
        iscale_new = cp.ones(rmax.shape, dtype=data.dtype)
        factor = cp.array([rrmin + rdelta * is_ + maxq - minq]).astype(data.dtype)[0]
        mask = rmin != rmax
        iscale_new[mask] = factor / (rmax[mask] - rmin[mask])
        quant_data_new = cp.clip(cp.round(iscale_new * (data - rmin)), minq, maxq)
        mul_weights_quant_data_new = weights * quant_data_new
        sum_l = cp.sum(mul_weights_quant_data_new, axis=1, keepdims=True)
        sum_l2 = cp.sum(mul_weights_quant_data_new * quant_data_new, axis=1, keepdims=True)
        sum_xl = cp.sum(mul_weights_quant_data_new * data, axis=1, keepdims=True)
        D = cp.subtract(sum_w * sum_l2, sum_l**2)  # noqa: N806

        this_scale = (sum_w * sum_xl - sum_x * sum_l) / D
        this_min = (sum_l2 * sum_x - sum_l * sum_xl) / D

        diff = this_scale * quant_data_new + this_min - data
        mad = cp.sum(weights * diff**2, axis=1, keepdims=True)

        mad_1 = cp.array(mad)
        best_mad_1 = cp.array(best_mad)
        idx_to_replace = cp.where(mad_1 < best_mad_1)[0]
        quant_data[idx_to_replace, :] = quant_data_new[idx_to_replace, :]
        best_mad[idx_to_replace] = mad[idx_to_replace]
        scale[idx_to_replace] = this_scale[idx_to_replace]
        rmin[idx_to_replace] = this_min[idx_to_replace]

    zero_point = cp.clip(((-rmin) / scale).round(), 0, maxq).astype("uint8")
    scale = scale.astype(cp.float64)
    q_weight = cp.empty_like(data, dtype=scale.dtype)
    cp.divide(data, scale, out=q_weight)
    cp.add(q_weight, zero_point, out=q_weight)
    cp.round(q_weight, out=q_weight)
    cp.clip(q_weight, minq, maxq, out=q_weight)

    return q_weight.get(), scale.get(), zero_point.get()


class OnnxKQuantQuantization(Pass):
    """Quantize ONNX models with the k-quant weight-only algorithm.

    K-quant uses weighted least-squares with iterative refinement to find
    optimal per-group scale and zero-point, achieving better accuracy than
    simple RTN (round-to-nearest) quantization at the same bit-width.

    Use ``customized_weight_config`` to assign per-node quantization settings
    (e.g., sensitive layers at INT8 while others use INT4).
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="Bits for weight-only quantization. Supports 4 or 8. Default value is 4.",
            ),
            "block_size": PassConfigParam(
                type_=int,
                default_value=32,
                description="Block size for quantization. Default value is 32.",
            ),
            "accuracy_level": PassConfigParam(
                type_=AccuracyLevel,
                default_value=AccuracyLevel.unset,
                description=(
                    "Accuracy level of the 4-bit quantized MatMul computation. Refer to the MatMulNBits"
                    " contrib op's 'accuracy_level' attribute for details"
                    " (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md"
                    "#commicrosoftmatmulnbits)."
                ),
            ),
            "customized_weight_config": PassConfigParam(
                type_=dict,
                default_value=None,
                description=(
                    "Per-node quantization overrides. A dict mapping node names to their config, "
                    'e.g. {"node_name": {"bits": 8, "group_size": 64}} to override bits or '
                    "group_size for specific nodes."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1
        self._quantize_model(
            ir_model,
            config.nodes_to_exclude,
            config.customized_weight_config,
            config.bits,
            config.block_size,
            config.accuracy_level,
        )
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    def _quantize_model(
        self,
        ir_model: ir.Model,
        nodes_to_exclude: Optional[list[str]] = None,
        customized_weight_config: Optional[dict] = None,
        bits: int = 4,
        block_size: int = 32,
        accuracy_level: AccuracyLevel = AccuracyLevel.unset,
    ):
        nodes_to_exclude = nodes_to_exclude or []
        customized_weight_config = customized_weight_config or {}

        globally_registered = {}

        ir_model.graph.sort()
        for node in ir_model.graph.all_nodes():
            node_name = node.name

            if node_name in nodes_to_exclude:
                logger.debug("Exclude quantization of %s as specified by nodes_to_exclude.", node_name)
                continue

            if node.op_type != str(OpType.MatMul):
                continue

            if not node.inputs[1].is_initializer():
                logger.debug("Skip quantization of %s: weight is not an initializer.", node_name)
                continue

            # Resolve per-node config overrides
            node_config = customized_weight_config.get(node_name, {})
            node_bits = node_config.get("bits", bits)
            node_block_size = node_config.get("group_size", block_size)

            quantized_node, initializer_graph = self._quantize_matmul(node, node_bits, node_block_size, accuracy_level)

            if quantized_node.op_type == OpType.MatMulNBits:
                registered = {}
                for input_value in quantized_node.inputs:
                    if input_value.const_value is not None:
                        if input_value.name in globally_registered:
                            ir.convenience.replace_all_uses_with(input_value, globally_registered[input_value.name])
                        elif input_value.name not in registered:
                            initializer_graph.register_initializer(input_value)
                            registered[input_value.name] = input_value
                            globally_registered[input_value.name] = input_value
                        else:
                            logger.debug(
                                "Found duplicated initializer %s, replace all uses with the first one.",
                                input_value.name,
                            )
                            ir.convenience.replace_all_uses_with(input_value, registered[input_value.name])

                ir.convenience.replace_nodes_and_values(
                    node.graph, node, [node], [quantized_node], node.outputs, quantized_node.outputs
                )

        # Remove orphaned initializers
        used_names: set[str] = set()
        for node in ir_model.graph.all_nodes():
            for inp in node.inputs:
                if inp is not None and inp.name:
                    used_names.add(inp.name)
        for out in ir_model.graph.outputs:
            if out is not None and out.name:
                used_names.add(out.name)

        unused = [name for name in ir_model.graph.initializers if name not in used_names]
        for name in unused:
            del ir_model.graph.initializers[name]
        if unused:
            logger.info("Removed %d unused initializers after quantization.", len(unused))

    def _quantize_matmul(
        self, node: ir.Node, bits: int, block_size: int, accuracy_level: AccuracyLevel
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight B of a MatMul node using the k-quant algorithm."""
        node_initializer = node.inputs[1]
        b_ndarray = node_initializer.const_value.numpy()

        if len(b_ndarray.shape) != 2:
            logger.debug("MatMul weight is not 2D. Skip quantization of %s.", node.name)
            return node, node.graph

        packed, scales, zero_point = self._kquant_block_quant(b_ndarray, bits, block_size)

        b_quant = ir.Value(name=node_initializer.name + f"_Q{bits}", const_value=ir.tensor(packed))
        scales_tensor = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))
        # K-quant is always asymmetric, so zero_point is always present
        zero_point_tensor = ir.Value(name=node_initializer.name + "_zero_point", const_value=ir.tensor(zero_point))
        node_inputs = [node.inputs[0], b_quant, scales_tensor, zero_point_tensor]

        rows, cols = b_ndarray.shape
        kwargs = {
            "K": rows,
            "N": cols,
            "bits": bits,
            "block_size": block_size,
        }
        if accuracy_level > 0:
            kwargs["accuracy_level"] = accuracy_level

        node.outputs[0].name = node.outputs[0].name + f"_Q{bits}"

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.MatMulNBits),
            inputs=node_inputs,
            name=node.name + f"_Q{bits}" if node.name else "",
            attributes=kwargs,
        ), node_initializer.graph

    @staticmethod
    def _kquant_block_quant(
        fp32weight: np.ndarray, bits: int, block_size: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Quantize a 2D weight matrix using the k-quant algorithm.

        Args:
            fp32weight: weight matrix, shape (rows, cols) = (K, N).
            bits: quantization bit-width (4 or 8).
            block_size: number of elements per quantization group.

        Returns:
            packed: packed quantized weight, shape (N, k_blocks, blob_size), dtype uint8.
            scales: per-group scales, shape (N * k_blocks,), dtype same as input.
            zero_point: packed zero points, dtype uint8.

        """
        rows, cols = fp32weight.shape
        if bits not in (4, 8):
            raise ValueError(f"MatMulNBits does not support num_bits = {bits}. Use 4 or 8.")

        k_blocks = (rows + block_size - 1) // block_size

        # Pad rows to be divisible by block_size
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

        # Transpose to (N, K_padded) so each row is one output channel
        weight_t = fp32weight.T

        # Run k-quant quantization: operates on (-1, group_size) blocks
        q_weight, scale, zp = _kquant_quantize(weight_t, bits, block_size)
        # q_weight: (N * k_blocks, block_size), scale: (N * k_blocks, 1), zp: (N * k_blocks, 1)

        q_weight = q_weight.astype("uint8")

        # Pack quantized weights into MatMulNBits blob format
        blob_size = block_size * bits // 8
        if bits == 4:
            q_weight_pairs = q_weight[:, ::2] | (q_weight[:, 1::2] << 4)
            packed = q_weight_pairs[:, :blob_size]
        else:
            packed = q_weight

        packed = np.reshape(packed, (cols, k_blocks, blob_size))

        # Format scales: (N, k_blocks) -> flatten to (N * k_blocks,)
        scale = np.reshape(scale, (cols, k_blocks)).astype(fp32weight.dtype)
        scales_flat = scale.flatten()

        # Pack zero points
        zp = zp.flatten()  # (N * k_blocks,)
        if bits == 4:
            # Pack pairs of 4-bit zero points into uint8
            zp_per_col = np.reshape(zp, (cols, k_blocks))
            packed_zp = np.full((cols, (k_blocks + 1) // 2), 136, dtype="uint8")  # default 0x88
            for col_idx in range(cols):
                for j in range(k_blocks):
                    byte_idx = j // 2
                    if j % 2 == 0:
                        packed_zp[col_idx, byte_idx] = (packed_zp[col_idx, byte_idx] & 0xF0) | (
                            zp_per_col[col_idx, j] & 0x0F
                        )
                    else:
                        packed_zp[col_idx, byte_idx] = (packed_zp[col_idx, byte_idx] & 0x0F) | (
                            (zp_per_col[col_idx, j] & 0x0F) << 4
                        )
            zero_point_flat = packed_zp.flatten()
        else:
            zero_point_flat = zp.astype("uint8")

        return packed, scales_flat, zero_point_flat
