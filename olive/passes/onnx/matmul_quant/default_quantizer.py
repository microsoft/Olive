import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
import onnx
from onnx import GraphProto, NodeProto, TensorProto

from olive.passes.onnx.matmul_quant.utils import MSFT_DOMAIN, Algorithm, OpType, WeightOnlyQuantConfig, get_initializer

logger = logging.getLogger(__name__)


class DefaultWeightOnlyQuantConfig(WeightOnlyQuantConfig):
    def __init__(
        self,
        bits: int = 4,
        block_size: int = 128,
        is_symmetric: bool = False,
        accuracy_level: Optional[int] = None,
        op_types_to_quantize: Optional[tuple[str, ...]] = None,
        quant_axes: Optional[tuple[tuple[str, int], ...]] = None,
    ):
        """This is a class for weight only affine quantization configuration.

        Args:
            block_size (int, optional):
                channel number in one block to execute an affine quantization iteration.
            is_symmetric (bool, optional):
                whether quantize weight symmetrically.
            accuracy_level (int, optional):
                Accuracy level of the 4-bit quantized MatMul computation.
                Refer to the MatMulNBits contrib op's 'accuracy_level' attribute for details.
                (https://github.com/microsoft/onnxruntime/blob/main/docs/ContribOperators.md#commicrosoftmatmulnbits)
            quant_format (QuantFormat{QOperator, QDQ}, optional):
                QOperator format quantizes the model with quantized operators directly.
                QDQ format quantize the model by inserting QuantizeLinear/DeQuantizeLinear on the tensor.
                Defaults to QuantFormat.QOperator.
            op_types_to_quantize (optional):
                set of operator types to quantize.
            quant_axes (dict[str, int], optional):
                op:axis, which axis to quantize for an op. Default {MatMul: 0, Gather: 1}

        """
        from onnxruntime.quantization import QuantFormat

        super().__init__(
            algorithm=Algorithm.DEFAULT,
            quant_format=QuantFormat.QOperator,
            op_types_to_quantize=op_types_to_quantize,
            quant_axes=quant_axes,
        )
        self.block_size = block_size
        self.is_symmetric = is_symmetric
        self.bits = bits
        self.accuracy_level = accuracy_level


class DefaultWeightOnlyQuantizer:
    def __init__(self, config: DefaultWeightOnlyQuantConfig):
        self.config = config

    def qbits_block_quant(self, fp32weight: npt.ArrayLike) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """4b/8b quantize fp32 weight to int4 using C++ kernels."""
        from onnxruntime.capi._pybind_state import quantize_matmul_4bits, quantize_qdq_matmul_4bits
        from onnxruntime.quantization import QuantFormat

        qbits = self.config.bits
        kpack = 8 // qbits
        if len(fp32weight.shape) != 2:
            raise ValueError("Current int4 block quantization only supports 2D tensors!")
        rows, cols = fp32weight.shape

        block_size = self.config.block_size
        k_blocks = (rows + block_size - 1) // block_size

        if self.config.quant_format == QuantFormat.QOperator:
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

                    quantize_matmul_8bits(
                        packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
                    )
                else:
                    raise ValueError("Default quantization of 8-bit weights is not supported for onnxruntime<=1.21.1")
            else:
                quantize_matmul_4bits(
                    packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
                )
        else:
            # TODO(xiaoyu): Use QDQ pass when it is implemented
            assert qbits == 4, "QDQ format only support 4 bits quantization"
            packed = np.zeros((rows * cols + 1) // 2, dtype="uint8")
            zero_point = np.zeros((cols * k_blocks + 1) // 2, dtype="uint8")
            scales = np.zeros((k_blocks, cols), dtype=fp32weight.dtype)
            quantize_qdq_matmul_4bits(
                packed, fp32weight, scales, zero_point, block_size, cols, rows, self.config.is_symmetric
            )

        return (packed, scales, zero_point)

    def quantize_matmul(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """Quantize weight B of MatMul node to int4 or int8.
        Currently only support 2D constant matrix and axis 0 blockwise quantization.
        """
        from onnxruntime.quantization import QuantFormat

        bits = self.config.bits
        if bits == 8:
            qtype = TensorProto.INT8 if self.config.is_symmetric else TensorProto.UINT8
        else:
            qtype = TensorProto.INT4 if self.config.is_symmetric else TensorProto.UINT4
        input_b = node.input[1]
        b_tensor, b_graph = get_initializer(input_b, graph_stack)
        if b_tensor is None:
            logger.info("MatMul doesn't have const weight. Skip to quantize")
            return [node]  # only care about constant weight

        b_ndarray = onnx.numpy_helper.to_array(b_tensor)
        if len(b_ndarray.shape) != 2:
            logger.info("MatMul weight is not 2D. Skip to quantize")
            return [node]  # can only process 2-D matrix

        packed, scales, zero_points = self.qbits_block_quant(b_ndarray)

        if self.config.quant_format == QuantFormat.QOperator:
            b_quant = onnx.numpy_helper.from_array(packed, b_tensor.name + f"_Q{bits}")
            scales_tensor = onnx.numpy_helper.from_array(scales, b_tensor.name + "_scales")
        else:
            b_quant = onnx.helper.make_tensor(
                b_tensor.name + f"_DQ_Q{bits}", qtype, b_ndarray.shape, packed.tobytes(), True
            )
            scales_tensor = onnx.numpy_helper.from_array(scales, b_tensor.name + "_DQ_scales")

        for graph_input in b_graph.input:
            if graph_input.name == input_b:
                b_graph.input.remove(graph_input)
                break

        b_graph.initializer.extend([b_quant, scales_tensor])

        output_nodes = []

        if self.config.quant_format == QuantFormat.QOperator:
            input_names = [node.input[0], b_quant.name, scales_tensor.name]
            if not self.config.is_symmetric:
                zp_tensor = onnx.numpy_helper.from_array(zero_points, b_tensor.name + "_zero_points")
                input_names.append(zp_tensor.name)
                b_graph.initializer.extend([zp_tensor])
            kwargs = {}
            rows, cols = b_ndarray.shape
            kwargs["K"] = rows
            kwargs["N"] = cols
            kwargs["bits"] = bits
            kwargs["block_size"] = self.config.block_size
            if self.config.accuracy_level is not None:
                kwargs["accuracy_level"] = self.config.accuracy_level

            matmul_qbit_node = onnx.helper.make_node(
                "MatMulNBits",
                inputs=input_names,
                outputs=[node.output[0]],
                name=node.name + f"_Q{bits}" if node.name else "",
                domain="com.microsoft",
                **kwargs,
            )

            output_nodes.append(matmul_qbit_node)
        else:
            dq_input_names = [b_quant.name, scales_tensor.name]
            dq_output_names = [b_quant.name + "_output"]
            matmul_input_names = [node.input[0], dq_output_names[0]]
            matmul_output_names = [node.output[0]]
            if not self.config.is_symmetric:
                zp_tensor = onnx.helper.make_tensor(
                    b_tensor.name + "_DQ_zero_points", qtype, scales.shape, zero_points.tobytes(), True
                )
                dq_input_names.append(zp_tensor.name)
                b_graph.initializer.extend([zp_tensor])
            dq_kwargs = {"axis": 0, "block_size": self.config.block_size}
            dq_node = onnx.helper.make_node(
                "DequantizeLinear",
                inputs=dq_input_names,
                outputs=dq_output_names,
                name=node.name + f"_DQ_Q{bits}" if node.name else "",
                **dq_kwargs,
            )
            matmul_node = onnx.helper.make_node(
                "MatMul",
                inputs=matmul_input_names,
                outputs=matmul_output_names,
                name=node.name + f"_matmul_Q{bits}" if node.name else "",
            )
            output_nodes.extend([dq_node, matmul_node])

        return output_nodes

    @staticmethod
    def quant_slice_symmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        max_val = np.max(data, axis=1, keepdims=True)
        min_val = np.min(data, axis=1, keepdims=True)
        abs_max = np.where(np.abs(max_val) > np.abs(min_val), max_val, min_val)

        scale = abs_max / -8.0  # if max == min, max may be clipped
        quantized_slice = np.where(scale == 0, 0, data / scale).round().clip(-8, 7).astype(np.int8)

        return quantized_slice, scale

    @staticmethod
    def quant_slice_asymmetric(data: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        min_val = np.minimum(data.min(axis=1, keepdims=True), 0)
        max_val = np.maximum(data.max(axis=1, keepdims=True), 0)

        scale = (max_val - min_val) / 15.0
        zero_point = np.where(scale == 0, 8, -min_val / scale).round().clip(0, 15).astype(np.uint8)
        quantized_slice = np.where(scale == 0, 8, data / scale + zero_point).round().clip(0, 15).astype(np.uint8)

        return quantized_slice, scale, zero_point

    @staticmethod
    def pack_int8_to_int4(data: np.ndarray) -> np.ndarray:
        """Pack int8 data to int4 and store in uint8 ndarray."""
        data_flat = data.reshape(-1)
        if len(data_flat) % 2 != 0:
            data_flat = np.append(data_flat, 0)
        quant_data_int4 = (data_flat[::2] & 0xF) | ((data_flat[1::2] & 0xF) << 4)

        return quant_data_int4.astype("uint8")

    @staticmethod
    def quantize_ndarray(
        data: np.ndarray,
        quantize_axis: int,
        block_size: int,
        is_symmetric: bool,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
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
                quantized_slice_int8, scale_slice = DefaultWeightOnlyQuantizer.quant_slice_symmetric(block_slice)
            else:
                quantized_slice_int8, scale_slice, zero_point_slice_int8 = (
                    DefaultWeightOnlyQuantizer.quant_slice_asymmetric(block_slice)
                )

            quant_data_int8[:, i:end_idx, :] = quantized_slice_int8
            j = i // block_size
            scales[:, j : (j + 1), :] = scale_slice
            if not is_symmetric:
                zero_point_int8[:, j : (j + 1), :] = zero_point_slice_int8

        # pack int8 to int4
        quant_data_int4 = DefaultWeightOnlyQuantizer.pack_int8_to_int4(quant_data_int8)
        zero_point_int4 = None
        if not is_symmetric:
            zero_point_int4 = DefaultWeightOnlyQuantizer.pack_int8_to_int4(zero_point_int8)
        scales = scales.reshape(scales_shape)
        return quant_data_int4, scales, zero_point_int4

    def quantize_gather(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """Quantize weight data of Gather node to int4."""
        from onnxruntime.quantization import QuantFormat

        assert self.config.quant_format == QuantFormat.QOperator, "Gather only supports QOperator format currently."

        qtype = TensorProto.INT4 if self.config.is_symmetric else TensorProto.UINT4
        data_arg = node.input[0]
        data_tensorproto, data_graphproto = get_initializer(data_arg, graph_stack)
        if data_tensorproto is None:
            logger.info("Gather doesn't have const weight. Skip quantization.")
            return [node]  # only care about constant weight

        data_ndarray = onnx.numpy_helper.to_array(data_tensorproto)
        data_rank = len(data_ndarray.shape)
        quantize_axis = self.config.quant_axes.get("Gather", 1)
        block_size = self.config.block_size

        assert -data_rank <= quantize_axis < data_rank, "Invalid quantize axis for Gather node."
        assert block_size >= 16 and ((block_size - 1) & block_size == 0), "Invalid block size for Gather node."

        quantize_axis = (quantize_axis + data_rank) % data_rank
        quantized_data, scales, zero_points = self.quantize_ndarray(
            data_ndarray, quantize_axis, block_size, self.config.is_symmetric
        )

        for graph_input in data_graphproto.input:
            if graph_input.name == data_arg:
                data_graphproto.input.remove(graph_input)
                break

        quantized_data_tensorproto = onnx.helper.make_tensor(
            data_tensorproto.name + "_Q4", qtype, data_ndarray.shape, quantized_data.tobytes(), True
        )
        scales_tensorproto = onnx.numpy_helper.from_array(scales, data_tensorproto.name + "_scales")
        input_names = [quantized_data_tensorproto.name, node.input[1], scales_tensorproto.name]
        data_graphproto.initializer.extend([quantized_data_tensorproto, scales_tensorproto])
        if not self.config.is_symmetric:
            zp_tensorproto = onnx.helper.make_tensor(
                data_tensorproto.name + "_zero_points", qtype, scales.shape, zero_points.tobytes(), True
            )
            input_names.append(zp_tensorproto.name)
            data_graphproto.initializer.extend([zp_tensorproto])

        try:
            gather_axis = onnx.helper.get_node_attr_value(node, "axis")
        except ValueError:
            gather_axis = 0

        kwargs = {
            "gather_axis": gather_axis,
            "quantize_axis": quantize_axis,
            "block_size": block_size,
        }

        gather_q4_node = onnx.helper.make_node(
            str(OpType.GatherBlockQuantized),
            inputs=input_names,
            outputs=[node.output[0]],
            name=node.name + "_Q4" if node.name else "",
            domain=str(MSFT_DOMAIN),
            **kwargs,
        )

        return [gather_q4_node]

    def quantize(self, node: NodeProto, graph_stack: list[GraphProto]) -> list[NodeProto]:
        """Target node:        QOperator node:            QDQ nodes:
        MatMul              MatMulNBits                DeQuantizeLinear -> MatMul
        Gather              GatherBlockQuantized       Gather, Gather, Gather (optional) -> DequantizeLinear
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        If QOperator format, return the corresponding QOperator nodes.
        If QDQ format, return the corresdponging QDQ nodes.
        Gather (quantized data) + Gather (scales) + Gather (optional, zero points) -> DequantizeLinear is
        not supported yet because Gather does not support int4 data.
        """
        from onnxruntime.quantization import QuantFormat

        logger.info("start to quantize %s ...", node.name)

        bits = self.config.bits
        if node.op_type == "MatMul":
            if bits == 8 and self.config.quant_format == QuantFormat.QDQ:
                logger.error("MatMul only supports QOperator format for 8 bits quantization.")
                return [node]
            results = self.quantize_matmul(node, graph_stack)
        elif node.op_type == "Gather":
            if self.config.bits != 4:
                logger.error("Gather only supports 4 bits quantization.")
                return [node]

            results = self.quantize_gather(node, graph_stack)
        else:
            logger.error("Unsupported operator %s for weight only quantization. Skip quantization.", node.op_type)
            return [node]

        logger.info("complete quantization of %s with %s bits ...", node.name, self.config.bits)
        return results
