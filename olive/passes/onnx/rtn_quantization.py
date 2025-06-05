# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import onnx
from onnx.onnx_pb import NodeProto

from olive.constants import MSFT_DOMAIN, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    model_has_adapters,
    model_proto_to_olive_model,
)
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxRtnQuantization(Pass):
    """Quantize ONNX models with HQQ algorithm."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "bits": PassConfigParam(
                type_=int,
                default_value=4,
                description="Bits for quantization. Default value is 4.",
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
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        dag = OnnxDAG(model.load_model())
        dag.set_opset_import(MSFT_DOMAIN, 1)
        dag = self._process_graph(dag, config.block_size, config.axis, config.nodes_to_exclude, config.nodes_to_include)
        dag.update()
        return model_proto_to_olive_model(dag.model, output_model_path, config)

    def _process_graph(
        self,
        dag: OnnxDAG,
        block_size: int = 128,
        axis: int = 0,
        nodes_to_exclude: Optional[list[str]] = None,
        nodes_to_include: Optional[list[str]] = None,
    ):
        node_quantizer = RtnQuantizer(bits=self.config.bits, block_size=block_size, axis=axis, is_symmetric=True)

        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_include = nodes_to_include or []

        ordered_nodes = dag.topological_sort()

        for node_name in ordered_nodes:
            node = dag.get_node(node_name)

            if node_name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node_name)
                continue

            elif node.op_type in (str(OpType.MatMul), str(OpType.Gather)) and (
                node_name in nodes_to_include or len(nodes_to_include) == 0
            ):
                graph_idx = dag.get_graph_idx(node_name)
                logger.debug("quantize node %s", node_name)
                node_initializers = dag.get_node_initializers(node_name)
                quantized_nodes, initializers = node_quantizer.quantize(node.proto, node_initializers)

                if quantized_nodes[0].op_type == str(OpType.MatMulNBits) or quantized_nodes[0].op_type == str(
                    OpType.GatherBlockQuantized
                ):
                    for initializer in initializers:
                        dag.add_initializer(initializer, graph_idx)
                    dag.add_node(quantized_nodes[0], graph_idx)
                    node_output = node.proto.output[0]
                    new_node_output = quantized_nodes[0].output[0]
                    for consumer in dag.get_consumers(node_name):
                        dag.replace_node_input(consumer, node_output, new_node_output)

                    is_model_output = dag.is_output(node_output)
                    original_proto = None

                    if is_model_output:
                        original_proto = dag.get_io(node_output).proto[0]
                        dag.remove_output(node_output)

                    dag.remove_node(node_name)

                    if is_model_output:
                        dag.rename_node_output(quantized_nodes[0].name, new_node_output, node_output)
                        vi = onnx.ValueInfoProto()
                        vi.CopyFrom(original_proto)
                        dag.get_io(node_output).proto = [vi]
                        dag.make_output(node_output)

            else:
                logger.debug("skip to quantize %s ...", node_name)
        return dag


class RtnQuantizer:
    """RTN Quantizer class for quantizing ONNX models."""

    def __init__(
        self,
        bits: int,
        block_size: int,
        axis: int,
        is_symmetric: bool,
    ):
        self.bits = bits
        self.block_size = block_size
        self.axis = axis
        self.is_symmetric = is_symmetric

    def qbits_block_quant(self, fp32weight: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4b/8b quantize fp32 weight to int4 using C++ kernels."""
        from onnxruntime.capi._pybind_state import quantize_matmul_4bits

        qbits = self.bits
        kpack = 8 // qbits
        if len(fp32weight.shape) != 2:
            raise ValueError("Current int4 block quantization only supports 2D tensors!")
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

                quantize_matmul_8bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, self.is_symmetric)
            else:
                raise ValueError("RTN quantization of 8-bit weights is not supported for onnxruntime<=1.21.1")
        else:
            quantize_matmul_4bits(packed, fp32weight, scales, zero_point, block_size, cols, rows, self.is_symmetric)

        return (packed, scales, zero_point)

    def quantize_matmul(self, node: NodeProto, node_initializers: list[onnx.TensorProto]) -> list[NodeProto]:
        """Quantize weight B of MatMul node to int4 or int8.

        Currently only support 2D constant matrix and axis 0 blockwise quantization.
        """
        if len(node_initializers) == 0:
            logger.debug("MatMul doesn't have const weight. Skip to quantize")
            return [node], []

        bits = self.bits
        b_pb = node_initializers[0]

        b_ndarray = onnx.numpy_helper.to_array(b_pb)
        if len(b_ndarray.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return [node], []  # can only process 2-D matrix

        packed, scales, zero_points = self.qbits_block_quant(b_ndarray)

        b_quant = onnx.numpy_helper.from_array(packed)
        b_quant.name = b_pb.name + f"_Q{bits}"
        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = b_pb.name + "_scales"
        zp_tensor = onnx.numpy_helper.from_array(zero_points)
        zp_tensor.name = b_pb.name + "_zero_points"

        output_nodes = []

        input_names = [node.input[0], b_quant.name, scales_tensor.name, zp_tensor.name]
        initializers = [b_quant, scales_tensor, zp_tensor]

        kwargs = {}
        rows, cols = b_ndarray.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = bits
        kwargs["block_size"] = self.block_size

        new_output = node.output[0] + "_Q4"

        matmul_qbit_node = onnx.helper.make_node(
            str(OpType.MatMulNBits),
            inputs=input_names,
            outputs=[new_output],
            name=node.name + f"_Q{bits}" if node.name else "",
            domain=MSFT_DOMAIN,
            **kwargs,
        )

        output_nodes.append(matmul_qbit_node)

        return output_nodes, initializers

    def quant_slice_symmetric(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)

        scale = abs_max / -8.0  # if max == min, max may be clipped
        quantized_slice = np.where(scale == 0, 0, data / scale).round().clip(-8, 7).astype(np.int8)

        return quantized_slice, scale

    def quant_slice_asymmetric(self, data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_val = np.minimum(data.min(axis=1, keepdims=True), 0)
        max_val = np.maximum(data.max(axis=1, keepdims=True), 0)

        scale = (max_val - min_val) / 15.0
        zero_point = np.where(scale == 0, 8, -min_val / scale).round().clip(0, 15).astype(np.uint8)
        quantized_slice = np.where(scale == 0, 8, data / scale + zero_point).round().clip(0, 15).astype(np.uint8)

        return quantized_slice, scale, zero_point

    def pack_int8_to_int4(self, data: np.ndarray) -> np.ndarray:
        """Pack int8 data to int4 and store in uint8 ndarray."""
        data_flat = data.reshape(-1)
        if len(data_flat) % 2 != 0:
            data_flat = np.append(data_flat, 0)
        quant_data_int4 = (data_flat[::2] & 0xF) | ((data_flat[1::2] & 0xF) << 4)

        return quant_data_int4.astype("uint8")

    def quantize_ndarray(
        self,
        data: np.ndarray,
        quantize_axis: int,
        block_size: int,
        is_symmetric: bool,
    ) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Quantize ndarray data to int4 using numpy, return (quantized data, scales, zero points)."""
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
            zero_point_slice_int8 = None

            if is_symmetric:
                quantized_slice_int8, scale_slice = self.quant_slice_symmetric(block_slice)
            else:
                quantized_slice_int8, scale_slice, zero_point_slice_int8 = self.quant_slice_asymmetric(block_slice)

            quant_data_int8[:, i:end_idx, :] = quantized_slice_int8
            j = i // block_size
            scales[:, j : (j + 1), :] = scale_slice
            if not is_symmetric:
                zero_point_int8[:, j : (j + 1), :] = zero_point_slice_int8

        # pack int8 to int4
        quant_data_int4 = self.pack_int8_to_int4(quant_data_int8)
        zero_point_int4 = None
        if not is_symmetric:
            zero_point_int4 = self.pack_int8_to_int4(zero_point_int8)
        scales = scales.reshape(scales_shape)
        return quant_data_int4, scales, zero_point_int4

    def quantize_gather(self, node: NodeProto, node_initializers: list[onnx.TensorProto]) -> list[NodeProto]:
        """Quantize weight data of Gather node to int4."""
        if len(node_initializers) == 0:
            logger.debug("Gather doesn't have const weight. Skip to quantize")
            return [node], []

        data_tensorproto = node_initializers[0]

        data_ndarray = onnx.numpy_helper.to_array(data_tensorproto)
        data_rank = len(data_ndarray.shape)
        quantize_axis = self.axis
        block_size = self.block_size

        assert -data_rank <= quantize_axis < data_rank, "Invalid quantize axis for Gather node."
        assert block_size >= 16, "Block size must be greater than or equal to 16."
        assert (block_size & (block_size - 1)) == 0, "Block size must be a power of 2."

        quantize_axis = (quantize_axis + data_rank) % data_rank
        quantized_data, scales, zero_points = self.quantize_ndarray(
            data_ndarray, quantize_axis, block_size, self.is_symmetric
        )

        quantized_data_tensorproto = onnx.numpy_helper.from_array(quantized_data)
        quantized_data_tensorproto.name = data_tensorproto.name + f"_Q{self.bits}"
        scales_tensorproto = onnx.numpy_helper.from_array(scales)
        scales_tensorproto.name = data_tensorproto.name + "_scales"
        input_names = [quantized_data_tensorproto.name, node.input[1], scales_tensorproto.name]
        initializers = [quantized_data_tensorproto, scales_tensorproto]
        if not self.is_symmetric:
            zp_tensorproto = onnx.numpy_helper.from_array(zero_points)
            zp_tensorproto.name = data_tensorproto.name + "_zero_points"
            input_names.append(zp_tensorproto.name)
            initializers.append(zp_tensorproto)

        try:
            gather_axis = onnx.helper.get_node_attr_value(node, "axis")
        except ValueError:
            gather_axis = 0

        kwargs = {
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
            "block_size": block_size,
        }

        new_output = node.output[0] + "_Q4"

        gather_q4_node = onnx.helper.make_node(
            str(OpType.GatherBlockQuantized),
            inputs=input_names,
            outputs=[new_output],
            name=node.name + "_Q4" if node.name else "",
            domain=MSFT_DOMAIN,
            **kwargs,
        )

        return [gather_q4_node], initializers

    def quantize(self, node: NodeProto, node_initializers: list[onnx.TensorProto]) -> list[NodeProto]:
        """Quantize the weight of the target node and return the new nodes.

        Target node:        QOperator node:
        MatMul              MatMulNBits
        Gather              GatherBlockQuantized
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        return the corresponding QOperator nodes.
        """
        logger.info("start to quantize %s ...", node.name)

        if node.op_type == str(OpType.MatMul):
            results, initializers = self.quantize_matmul(node, node_initializers)
        elif node.op_type == str(OpType.Gather):
            if self.bits != 4:
                logger.error("Gather only supports 4 bits quantization.")
                return [node], []

            results, initializers = self.quantize_gather(node, node_initializers)
        else:
            logger.error("Unsupported operator %s for weight only quantization. Skip quantization.", node.op_type)
            return [node], []

        logger.info("complete quantization of %s with %s bits ...", node.name, self.bits)
        return results, initializers
