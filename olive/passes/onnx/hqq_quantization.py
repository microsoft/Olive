# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

import onnx
import torch

from olive.constants import MSFT_DOMAIN, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    model_has_adapters,
    model_proto_to_olive_model,
)
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxHqqQuantization(Pass):
    """Quantize ONNX models with HQQ algorithm."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
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
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        if model_has_adapters(model.model_path):
            logger.info(
                "HQQ quantization is not supported for models with adapters. Returning the model without quantization."
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
        node_quantizer = HqqQuantizer(block_size=block_size, axis=axis)

        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_include = nodes_to_include or []

        ordered_nodes = dag.topological_sort()

        for node_name in ordered_nodes:
            node = dag.get_node(node_name)

            if node_name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node_name)
                continue

            elif node.op_type == str(OpType.MatMul) and (node_name in nodes_to_include or len(nodes_to_include) == 0):
                graph_idx = dag.get_graph_idx(node_name)
                logger.debug("quantize node %s", node_name)
                node_initializers = dag.get_node_initializers(node_name)
                if len(node_initializers) == 0:
                    logger.debug("MatMul doesn't have const weight. Skip to quantize")
                    continue
                quantized_nodes, initializers = node_quantizer.quantize(node.proto, node_initializers)

                if quantized_nodes[0].op_type == str(OpType.MatMulNBits):
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


class HqqQuantizer:
    """HQQ Quantizer class for quantizing ONNX models."""

    def __init__(
        self,
        block_size: int,
        axis: int,
    ):
        self.block_size = block_size
        self.axis = axis

    # Proximal solver || weight - dequantize(quantize(weight))||_p^p
    @staticmethod
    def optimize_weights(tensor, scale, zero, min_max: list[int], axis: int = 0):
        lp_norm = 0.7
        beta = 1e1
        kappa = 1.01
        iters = 20

        dtype = torch.float16 if tensor.is_cuda else torch.float32
        w_f = tensor.to(dtype)
        scale = scale.to(dtype)
        zero = zero.to(dtype)

        def shrink_op(x, beta, p=lp_norm):
            if p == 1:
                # Formula: sign(x) * max(abs(x) - 1/beta, 0)
                return torch.sign(x) * torch.nn.functional.relu(torch.abs(x) - 1.0 / beta)
            else:
                # Formula: sign(x) * max(abs(x) - (1/beta) * abs(x)^(p-1), 0)
                return torch.sign(x) * torch.nn.functional.relu(
                    torch.abs(x) - (1.0 / beta) * torch.pow(torch.abs(x) + 1e-8, p - 1)
                )

        best_error = 1e4
        for _ in range(iters):
            w_q = torch.round(w_f * scale + zero).clamp(min_max[0], min_max[1])
            w_r = (w_q - zero) / scale
            w_e = shrink_op(w_f - w_r, beta)
            zero = torch.mean(w_q - (w_f - w_e) * scale, axis=axis, keepdim=True)
            beta *= kappa

            current_error = float(torch.abs(w_f - w_r).mean())
            if current_error < best_error:
                best_error = current_error
            else:
                break

        del w_f, w_q, w_r, w_e

        return scale, zero

    @staticmethod
    def pack_on_row_fast_248bit(pack_tensor, ori_int_tensor):
        if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
            ori_int_tensor = ori_int_tensor.T
            pack_tensor = pack_tensor.T
        compress_ratio = pack_tensor.element_size() * 8 // 4  # 4 bits
        for j in range(compress_ratio):
            pack_tensor[0:] |= ori_int_tensor[j::compress_ratio] << (4 * (j))  # 4 bits

    # from Official implementation of Half-Quadratic Quantization (HQQ)
    def quantize_internal(self, tensor, channel_wise=True, group_size=64, optimize=True, round_zero=True, axis=1):
        bits = 4  # 4 bits
        weight = tensor.float()
        ori_shape = weight.shape

        pad_len = (group_size - ori_shape[axis] % group_size) % group_size
        if axis == 1:
            weight = torch.nn.functional.pad(weight, (0, pad_len), "constant", 0)
        else:
            weight = torch.nn.functional.pad(weight, (0, 0, 0, pad_len), "constant", 0)
        shape = weight.shape

        # Reshape for grouping
        if (group_size is not None) and channel_wise:
            weight = weight.reshape([-1, group_size]) if (axis == 1) else weight.reshape([group_size, -1])

        # Get min/max values
        if channel_wise is False:
            _min, _max = weight.min(), weight.max()
            optimize = False
        else:
            _min = weight.min(axis=axis, keepdim=True)[0]
            _max = weight.max(axis=axis, keepdim=True)[0]

        max_v = 2**bits - 1
        min_v = 0
        min_max = [min_v, max_v]

        # Note: here we work with the inverse of the scale to avoid division and quantize instead via weight*scale + zero, the scale is inverted later on.
        # clamp to avoid half-precision problems
        scale = (max_v / (_max - _min)).clamp(max=2e4)
        min_max_axis = _max - _min
        if (min_max_axis == 0).sum().item() > 0:
            min_max_axis[min_max_axis == 0] = max_v
            scale = (max_v / min_max_axis).clamp(max=2e4)
        zero = -_min * scale

        if round_zero:
            zero = torch.round(zero)

        # Fine-tune weights
        if optimize:
            scale, zero = self.optimize_weights(tensor=weight, scale=scale, zero=zero, min_max=min_max, axis=axis)

        # Quantize
        # Necessary for fake quantization backprop
        w_q = torch.round(weight * scale + zero).clamp(min_max[0], min_max[1])
        w_q = w_q.reshape(shape).int()

        scale = 1.0 / scale
        if axis == 1:
            scale = scale.reshape(shape[0], -1)
            zero = zero.reshape(shape[0], -1)
        else:
            scale = scale.reshape(-1, shape[-1])
            zero = zero.reshape(-1, shape[-1])
        # cleanup
        del weight, _min, _max

        return w_q, scale.to(tensor.dtype), zero.to(tensor.dtype)

    def quantize(self, node: onnx.NodeProto, node_initializers: list[onnx.TensorProto]) -> list[onnx.NodeProto]:
        """Quantize the weight of the target node and return the new nodes.

        Target node:        QOperator node:
        MatMul              MatMulNBits
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes and initializers.
        """
        logger.debug("start to quantize %s ...", node.name)
        b_pb = node_initializers[0]

        b_array = onnx.numpy_helper.to_array(b_pb)
        if len(b_array.shape) != 2:
            logger.debug("MatMul weight is not 2D. Skip to quantize")
            return [node]  # can only process 2-D matrix
        b_array_torch = torch.from_numpy(b_array)
        if torch.cuda.is_available():
            b_array_torch = b_array_torch.cuda()
        quant_weight_torch, scales_torch, zero_points_torch = self.quantize_internal(
            b_array_torch.T, group_size=self.block_size
        )
        quant_weight_torch = quant_weight_torch.contiguous()
        scales_torch = scales_torch.contiguous()
        zero_points_torch = zero_points_torch.contiguous()

        packed_torch = torch.zeros(
            (quant_weight_torch.shape[0], quant_weight_torch.shape[1] // 2),
            dtype=torch.uint8,
            device=quant_weight_torch.device,
        )
        self.pack_on_row_fast_248bit(packed_torch, quant_weight_torch)
        scales = scales_torch.cpu().numpy()
        zero_points = zero_points_torch.cpu().numpy()
        # reshape to the predefined shape in MatmulNbits
        scales = scales.reshape(-1)
        zero_points = zero_points.reshape(-1)
        rows, cols = b_array_torch.shape
        block_size = self.block_size
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        packed_torch = packed_torch.reshape(cols, k_blocks, blob_size)

        b_quant = onnx.numpy_helper.from_array(packed_torch.cpu().numpy())
        b_quant.name = b_pb.name + "_Q4"
        scales_tensor = onnx.numpy_helper.from_array(scales)
        scales_tensor.name = b_pb.name + "_scales"
        zp_tensor = onnx.numpy_helper.from_array(zero_points)
        zp_tensor.name = b_pb.name + "_zero_points"

        input_names = [node.input[0], b_quant.name, scales_tensor.name, zp_tensor.name]
        initializers = [b_quant, scales_tensor, zp_tensor]

        kwargs = {}
        rows, cols = b_array.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = 4  # 4 bits
        kwargs["block_size"] = self.block_size

        new_output = node.output[0] + "_Q4"

        matmul_q4_node = onnx.helper.make_node(
            str(OpType.MatMulNBits),
            inputs=input_names,
            outputs=[new_output],
            name=node.name + "_Q4" if node.name else "",
            domain=MSFT_DOMAIN,
            **kwargs,
        )

        logger.debug("complete quantization of %s ...", node.name)

        return [matmul_q4_node], initializers
