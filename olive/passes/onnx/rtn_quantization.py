# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
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


class OnnxBlockWiseRtnQuantization(Pass):
    """Quantize ONNX models with weight-only block-wise RTN algorithm."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="Bits for weight-only quantization. Default value is 4.",
            ),
            "block_size": PassConfigParam(
                type_=int,
                default_value=128,
                description="Block size for quantization. Default value is 128.",
            ),
            "axis": PassConfigParam(
                type_=int,
                default_value=0,
                description="Axis to quantize. Default value is 0.",
            ),
            "is_symmetric": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Whether to use symmetric quantization. Default value is True.",
            ),
            "accuracy_level": PassConfigParam(
                type_=AccuracyLevel,
                default_value=AccuracyLevel.unset,
                description=(
                    "Accuracy level of the 4-bit quantized MatMul computation. Refer to the MatMulNBits contrib op's"
                    " 'accuracy_level' attribute for details"
                    " (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits)."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
            "nodes_to_include": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to include in quantization.",
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
            config.nodes_to_include,
            config.bits,
            config.block_size,
            config.axis,
            config.is_symmetric,
            config.accuracy_level,
        )
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    def _quantize_model(
        self,
        ir_model: ir.Model,
        nodes_to_exclude: Optional[list[str]] = None,
        nodes_to_include: Optional[list[str]] = None,
        bits: int = 4,
        block_size: int = 128,
        axis: int = 0,
        is_symmetric: bool = True,
        accuracy_level: AccuracyLevel = AccuracyLevel.unset,
    ):
        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_include = nodes_to_include or []

        # Track initializer names already registered across all nodes
        # to handle shared weights (e.g., pos_embed used by multiple Gather nodes).
        globally_registered = {}

        ir_model.graph.sort()
        for node in ir_model.graph.all_nodes():
            node_name = node.name

            if node_name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node_name)
                continue

            elif node.op_type in (str(OpType.MatMul), str(OpType.Gather)) and (
                node_name in nodes_to_include or not nodes_to_include
            ):
                # MatMul weight is inputs[1], Gather weight (embedding table) is inputs[0]
                weight_idx = 1 if node.op_type == str(OpType.MatMul) else 0
                if not node.inputs[weight_idx].is_initializer():
                    logger.debug("skip to quantize %s as it has no initializer", node_name)
                    continue

                if node.op_type == str(OpType.Gather) and bits not in (4, 8):
                    logger.warning(
                        "Gather quantization is only implemented for 4-bit and 8-bit. Skip node %s (bits=%d).",
                        node_name,
                        bits,
                    )
                    continue

                quantized_node, initializer_graph = self._quantize(
                    node, bits, block_size, axis, accuracy_level, is_symmetric
                )

                if quantized_node.op_type in (OpType.MatMulNBits, OpType.GatherBlockQuantized):
                    registered = {}
                    for input_value in quantized_node.inputs:
                        if input_value.const_value is not None:
                            if input_value.name in globally_registered:
                                # Already registered by a previous node (shared weight),
                                # replace with the existing initializer.
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
            else:
                logger.debug("skip to quantize %s ...", node_name)

        # Remove initializers that are no longer referenced by any node.
        # After quantization, the original FP32 weight initializers become orphaned
        # because the old MatMul/Gather nodes were replaced with MatMulNBits/GatherBlockQuantized
        # nodes that reference new INT4 weight initializers instead.
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

    def _quantize(
        self, node: ir.Node, bits: int, block_size: int, axis: int, accuracy_level: AccuracyLevel, is_symmetric: bool
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize the weight of the target node and return the new nodes.

        Target node:        QOperator node:
        MatMul              MatMulNBits
        Gather              GatherBlockQuantized
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        return the corresponding QOperator nodes.
        """
        logger.debug("Start quantizing %s ...", node.name)

        if node.op_type == OpType.MatMul:
            return self._quantize_matmul(node, bits, block_size, accuracy_level, is_symmetric)
        elif node.op_type == OpType.Gather:
            return self._quantize_gather(node, bits, block_size, axis, is_symmetric)
        else:
            logger.error("Unsupported op %s for weight-only quantization.", node.op_type)
            return node, node.graph

    def _quantize_matmul(
        self, node: ir.Node, bits: int, block_size: int, accuracy_level: AccuracyLevel, is_symmetric: bool
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight B of MatMul node to int4 or int8.

        Currently only support 2D constant matrix blockwise quantization.
        """
        node_initializer = node.inputs[1]
        b_ndarray = node_initializer.const_value.numpy()

        if len(b_ndarray.shape) != 2:
            logger.debug("MatMul weight is not 2D. Skip to quantize")
            return node, node.graph  # can only process 2-D matrix

        packed, scales, zero_point = self._qbits_block_quant(b_ndarray, bits, block_size, is_symmetric)

        b_quant = ir.Value(name=node_initializer.name + f"_Q{bits}", const_value=ir.tensor(packed))
        scales_tensor = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))
        if not is_symmetric:
            zero_point_tensor = ir.Value(name=node_initializer.name + "_zero_point", const_value=ir.tensor(zero_point))
            node_inputs = [node.inputs[0], b_quant, scales_tensor, zero_point_tensor]
        else:
            node_inputs = [node.inputs[0], b_quant, scales_tensor]

        kwargs = {}
        rows, cols = b_ndarray.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = bits
        kwargs["block_size"] = block_size
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

    def _quantize_gather(
        self, node: ir.Node, bits: int, block_size: int, axis: int, is_symmetric: bool
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight data of Gather node to int4."""
        node_initializer = node.inputs[0]
        data_ndarray = node_initializer.const_value.numpy()
        data_rank = len(data_ndarray.shape)

        # ORT GatherBlockQuantized requires quantize_axis == last dimension
        # when data is packed as uint8 (two int4 values per byte).
        quantize_axis = data_rank - 1

        assert -data_rank <= quantize_axis < data_rank, "Invalid quantize axis for Gather node."
        assert block_size >= 16, "Block size must be greater than or equal to 16."
        assert (block_size & (block_size - 1)) == 0, "Block size must be a power of 2."

        quantize_axis = (quantize_axis + data_rank) % data_rank
        quantized_data, scales, zero_point = self._quantize_ndarray(
            data_ndarray, quantize_axis, block_size, is_symmetric, bits
        )

        quantized_data_tensorproto = ir.Value(
            name=node_initializer.name + f"_Q{bits}", const_value=ir.tensor(quantized_data)
        )
        scales_tensorproto = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))
        if not is_symmetric:
            zero_point_tensorproto = ir.Value(
                name=node_initializer.name + "_zero_point", const_value=ir.tensor(zero_point)
            )
            node_inputs = [quantized_data_tensorproto, node.inputs[1], scales_tensorproto, zero_point_tensorproto]
        else:
            node_inputs = [quantized_data_tensorproto, node.inputs[1], scales_tensorproto]

        gather_axis = node.attributes.get_int("axis", 0)

        kwargs = {
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
            "block_size": block_size,
            "bits": bits,
        }

        node.outputs[0].name = node.outputs[0].name + f"_Q{bits}"

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.GatherBlockQuantized),
            inputs=node_inputs,
            name=node.name + f"_Q{bits}" if node.name else "",
            attributes=kwargs,
        ), node_initializer.graph

    def _qbits_block_quant(
        self, fp32weight: npt.ArrayLike, bits: int, block_size: int, is_symmetric: bool
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4b/8b quantize 2D fp32 weight to int4 using C++ kernels."""
        qbits = bits
        kpack = 8 // qbits
        rows, cols = fp32weight.shape

        k_blocks = (rows + block_size - 1) // block_size

        blob_size = (block_size + kpack - 1) // kpack
        padded_rows = k_blocks * block_size
        pad_len = padded_rows - rows
        if pad_len > 0:
            fp32weight = np.pad(fp32weight, ((0, pad_len), (0, 0)), "constant")

        # block wise quantization, each block comes from a single column
        packed = np.zeros((cols, k_blocks, blob_size), dtype="uint8")
        zero_point = np.zeros(cols * ((k_blocks + kpack - 1) // kpack), dtype="uint8")
        scales = np.zeros((cols * k_blocks), dtype=fp32weight.dtype)
        if qbits == 8:
            try:
                from onnxruntime import __version__ as OrtVersion
                from packaging import version
            except ImportError as e:
                raise ImportError("onnxruntime is required for RTN quantization.") from e

            if version.parse(OrtVersion) < version.parse("1.22.0"):
                raise ValueError("RTN quantization of 8-bit weights is not supported for onnxruntime<1.22.0")
            else:
                from onnxruntime.capi._pybind_state import quantize_matmul_8bits

                quantize_matmul_8bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, is_symmetric)
        else:
            try:
                from onnxruntime.capi._pybind_state import quantize_matmul_4bits
            except ImportError as e:
                raise ImportError("onnxruntime is required for RTN quantization.") from e

            quantize_matmul_4bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, is_symmetric)

        return (packed, scales, zero_point)

    @staticmethod
    def _quant_slice_symmetric(data: np.ndarray, bits: int = 4) -> tuple[np.ndarray, np.ndarray]:
        qmin = -(1 << (bits - 1))  # -8 for 4-bit, -128 for 8-bit
        qmax = (1 << (bits - 1)) - 1  # 7 for 4-bit, 127 for 8-bit
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)

        scale = abs_max / float(qmin)  # if max == min, max may be clipped
        quantized_slice = np.where(scale == 0, 0, data / scale).round().clip(qmin, qmax).astype(np.int8)

        return quantized_slice, scale

    @staticmethod
    def _quant_slice_asymmetric(data: np.ndarray, bits: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        qmax = (1 << bits) - 1  # 15 for 4-bit, 255 for 8-bit
        mid = 1 << (bits - 1)  # 8 for 4-bit, 128 for 8-bit
        min_val = np.minimum(data.min(axis=1, keepdims=True), 0)
        max_val = np.maximum(data.max(axis=1, keepdims=True), 0)

        scale = (max_val - min_val) / float(qmax)
        zero_point = np.where(scale == 0, mid, -min_val / scale).round().clip(0, qmax).astype(np.uint8)
        quantized_slice = np.where(scale == 0, mid, data / scale + zero_point).round().clip(0, qmax).astype(np.uint8)

        return quantized_slice, scale, zero_point

    @staticmethod
    def _pack_int4_along_axis(data: np.ndarray, axis: int = 1) -> np.ndarray:
        """Pack pairs of int4 values into uint8 along the specified axis.

        Unlike a flat pack, this correctly handles cases where the packing dimension is small
        (e.g., zero_point with k_blocks=1) by only pairing values within the same axis slice.
        """
        k = data.shape[axis]
        if k % 2 != 0:
            pad_width = [(0, 0)] * len(data.shape)
            pad_width[axis] = (0, 1)
            data = np.pad(data, pad_width)
        low = np.take(data, range(0, data.shape[axis], 2), axis=axis)
        high = np.take(data, range(1, data.shape[axis], 2), axis=axis)
        return ((low & 0xF) | ((high & 0xF) << 4)).astype("uint8")

    def _quantize_ndarray(
        self,
        data: np.ndarray,
        quantize_axis: int,
        block_size: int,
        is_symmetric: bool,
        bits: int = 4,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Quantize ndarray data to int4/int8 using numpy, return (quantized data, scales)."""
        # Get the shape of the matrix
        m = 1  # dimension of the matrix before the quantize axis
        k = data.shape[quantize_axis]  # dimension of the matrix along the quantize axis
        n = 1  # dimension of the matrix after the quantize axis
        for i, dim in enumerate(data.shape):
            if i < quantize_axis:
                m *= dim
            elif i > quantize_axis:
                n *= dim

        k_blocks = (k + block_size - 1) // block_size
        scales_shape = list(data.shape)
        scales_shape[quantize_axis] = k_blocks

        data_reshape = data.reshape((m, k, n))
        scales = np.zeros((m, k_blocks, n), dtype=data.dtype)
        zero_point_int8 = None
        if is_symmetric:
            quant_data_int8 = np.zeros((m, k, n), dtype="int8")
        else:
            quant_data_int8 = np.zeros((m, k, n), dtype="uint8")
            zero_point_int8 = np.zeros((m, k_blocks, n), dtype="uint8")

        # slice and quantize
        for i in range(0, k, block_size):
            end_idx = min(i + block_size, k)
            block_slice = data_reshape[:, i:end_idx, :]
            zero_point_slice = None
            if is_symmetric:
                quantized_slice_int8, scale_slice = self._quant_slice_symmetric(block_slice, bits)
            else:
                quantized_slice_int8, scale_slice, zero_point_slice = self._quant_slice_asymmetric(block_slice, bits)

            quant_data_int8[:, i:end_idx, :] = quantized_slice_int8
            j = i // block_size
            scales[:, j : (j + 1), :] = scale_slice
            if not is_symmetric:
                zero_point_int8[:, j : (j + 1), :] = zero_point_slice

        scales = scales.reshape(scales_shape)

        if bits <= 4:
            # pack int8 to int4
            # ORT GatherBlockQuantized uses unsigned int4 representation [0, 15]
            # where zero_point=8 is implied for symmetric quantization.
            # Convert signed int8 [-8, 7] to unsigned [0, 15] by adding 8.
            if is_symmetric:
                quant_data_int8 = (quant_data_int8.astype(np.int16) + 8).astype(np.uint8)

            # Pack along axis=1 (the quantize_axis in the 3D view: m, k, n).
            # This ensures packing pairs values within the same row, not across rows.
            quant_data_int4 = self._pack_int4_along_axis(quant_data_int8, axis=1)
            zero_point_int4 = None
            if not is_symmetric:
                zero_point_int4 = self._pack_int4_along_axis(zero_point_int8, axis=1)

            # Reshape packed data to match original rank (GatherBlockQuantized requires rank > 1).
            packed_shape = list(data.shape)
            packed_shape[quantize_axis] = (packed_shape[quantize_axis] + 1) // 2
            quant_data_int4 = quant_data_int4.reshape(packed_shape)
            if zero_point_int4 is not None:
                zp_shape = list(scales_shape)
                zp_shape[quantize_axis] = (zp_shape[quantize_axis] + 1) // 2
                zero_point_int4 = zero_point_int4.reshape(zp_shape)

            return quant_data_int4, scales, zero_point_int4
        else:
            # 8-bit: no packing needed, one value per byte.
            # Convert signed int8 [-128, 127] to unsigned uint8 [0, 255] by adding 128.
            if is_symmetric:
                quant_data_uint8 = (quant_data_int8.astype(np.int16) + 128).astype(np.uint8)
            else:
                quant_data_uint8 = quant_data_int8  # already uint8

            quant_data_uint8 = quant_data_uint8.reshape(data.shape)
            zero_point_out = None
            if not is_symmetric:
                zero_point_out = zero_point_int8.reshape(scales_shape)

            return quant_data_uint8, scales, zero_point_out
