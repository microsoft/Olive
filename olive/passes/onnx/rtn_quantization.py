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

from olive.constants import MSFT_DOMAIN, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    ir_model_to_olive_model,
    model_has_adapters,
)
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

# pylint: disable=W0201


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
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info(
                "RTN quantization is not supported for models with adapters. Returning the model without quantization."
            )
            return model
        self.bits = config.bits
        self.block_size = config.block_size
        self.axis = config.axis

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        ir_model = model.load_ir_model()
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1
        self._quantize_model(ir_model, config.nodes_to_exclude, config.nodes_to_include)
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    def _quantize_model(
        self,
        ir_model: ir.Model,
        nodes_to_exclude: Optional[list[str]] = None,
        nodes_to_include: Optional[list[str]] = None,
    ):
        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_include = nodes_to_include or []

        ir_model.graph.sort()
        for node in ir.traversal.RecursiveGraphIterator(ir_model.graph):
            node_name = node.name

            if node_name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node_name)
                continue

            elif node.op_type in (str(OpType.MatMul), str(OpType.Gather)) and (
                node_name in nodes_to_include or not nodes_to_include
            ):
                if (node.op_type == OpType.MatMul and not node.inputs[1].is_initializer()) or (
                    node.op_type == OpType.Gather and not node.inputs[0].is_initializer()
                ):
                    logger.debug("skip to quantize %s as it has no initializer", node_name)
                    continue

                if node.op_type == OpType.Gather and self.bits != 4:
                    logger.warning("Gather only supports 4-bit quantization.")
                    continue

                quantized_node, initializer_graph = self._quantize(node)

                if quantized_node.op_type in (OpType.MatMulNBits, OpType.GatherBlockQuantized):
                    registered = {}
                    for input_value in quantized_node.inputs:
                        if input_value.const_value is not None:
                            if input_value.name not in registered:
                                initializer_graph.register_initializer(input_value)
                                registered[input_value.name] = input_value
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

    def _quantize(self, node: ir.Node) -> tuple[ir.Node, ir.Graph]:
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
            return self._quantize_matmul(node)
        elif node.op_type == OpType.Gather:
            return self._quantize_gather(node)
        else:
            logger.error("Unsupported op %s for weight-only quantization.", node.op_type)
            return node, node.graph

    def _quantize_matmul(self, node: ir.Node) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight B of MatMul node to int4 or int8.

        Currently only support 2D constant matrix and axis 0 blockwise quantization.
        """
        node_initializer = node.inputs[1]
        bits = self.bits
        b_ndarray = node_initializer.const_value.numpy()

        if len(b_ndarray.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return node, node.graph  # can only process 2-D matrix

        packed, scales = self._qbits_block_quant(b_ndarray)

        b_quant = ir.Value(name=node_initializer.name + f"_Q{bits}", const_value=ir.tensor(packed))
        scales_tensor = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))

        node_inputs = [node.inputs[0], b_quant, scales_tensor]

        kwargs = {}
        rows, cols = b_ndarray.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = bits
        kwargs["block_size"] = self.block_size

        node.outputs[0].name = node.outputs[0].name + f"_Q{bits}"

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.MatMulNBits),
            inputs=node_inputs,
            name=node.name + f"_Q{bits}" if node.name else "",
            attributes=kwargs,
        ), node_initializer.graph

    def _quantize_gather(self, node: ir.Node) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight data of Gather node to int4."""
        node_initializer = node.inputs[0]
        data_ndarray = node_initializer.const_value.numpy()
        data_rank = len(data_ndarray.shape)
        quantize_axis = self.axis
        block_size = self.block_size

        assert -data_rank <= quantize_axis < data_rank, "Invalid quantize axis for Gather node."
        assert block_size >= 16, "Block size must be greater than or equal to 16."
        assert (block_size & (block_size - 1)) == 0, "Block size must be a power of 2."

        quantize_axis = (quantize_axis + data_rank) % data_rank
        quantized_data, scales = self._quantize_ndarray(data_ndarray, quantize_axis, block_size)

        quantized_data_tensorproto = ir.Value(
            name=node_initializer.name + f"_Q{self.bits}", const_value=ir.tensor(quantized_data)
        )
        scales_tensorproto = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))
        node_inputs = [quantized_data_tensorproto, node.inputs[1], scales_tensorproto]

        gather_axis = node.attributes.get_int("axis", 0)

        kwargs = {
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
            "block_size": block_size,
        }

        node.outputs[0].name = node.outputs[0].name + f"_Q{self.bits}"

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.GatherBlockQuantized),
            inputs=node_inputs,
            name=node.name + f"_Q{self.bits}" if node.name else "",
            attributes=kwargs,
        ), node_initializer.graph

    def _qbits_block_quant(self, fp32weight: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """4b/8b quantize 2D fp32 weight to int4 using C++ kernels."""
        from onnxruntime.capi._pybind_state import quantize_matmul_4bits

        qbits = self.bits
        kpack = 8 // qbits
        is_symmetric = True
        rows, cols = fp32weight.shape

        block_size = self.block_size
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
            from onnxruntime import __version__ as OrtVersion
            from packaging import version

            if version.parse(OrtVersion) > version.parse("1.21.0"):
                from onnxruntime.capi._pybind_state import quantize_matmul_8bits

                quantize_matmul_8bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, is_symmetric)
            else:
                raise ValueError("RTN quantization of 8-bit weights is not supported for onnxruntime<=1.21.1")
        else:
            quantize_matmul_4bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, is_symmetric)

        return (packed, scales)

    @staticmethod
    def _quant_slice_symmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)

        scale = abs_max / -8.0  # if max == min, max may be clipped
        quantized_slice = np.where(scale == 0, 0, data / scale).round().clip(-8, 7).astype(np.int8)

        return quantized_slice, scale

    @staticmethod
    def _pack_int8_to_int4(data: np.ndarray) -> np.ndarray:
        """Pack int8 data to int4 and store in uint8 ndarray."""
        data_flat = data.reshape(-1)
        if len(data_flat) % 2 != 0:
            data_flat = np.append(data_flat, 0)
        quant_data_int4 = (data_flat[::2] & 0xF) | ((data_flat[1::2] & 0xF) << 4)

        return quant_data_int4.astype("uint8")

    def _quantize_ndarray(
        self,
        data: np.ndarray,
        quantize_axis: int,
        block_size: int,
    ) -> Optional[tuple[np.ndarray, np.ndarray]]:
        """Quantize ndarray data to int4 using numpy, return (quantized data, scales)."""
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
        quant_data_int8 = np.zeros((m, k, n), dtype="int8")

        # slice and quantize
        for i in range(0, k, block_size):
            end_idx = min(i + block_size, k)
            block_slice = data_reshape[:, i:end_idx, :]
            quantized_slice_int8, scale_slice = self._quant_slice_symmetric(block_slice)

            quant_data_int8[:, i:end_idx, :] = quantized_slice_int8
            j = i // block_size
            scales[:, j : (j + 1), :] = scale_slice

        # pack int8 to int4
        quant_data_int4 = self._pack_int8_to_int4(quant_data_int8)
        scales = scales.reshape(scales_shape)
        return quant_data_int4, scales
