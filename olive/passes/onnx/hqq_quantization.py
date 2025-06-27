# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

import onnx_ir as ir
import torch

from olive.constants import MSFT_DOMAIN, AccuracyLevel, OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import (
    get_external_data_config,
    ir_model_to_olive_model,
    model_has_adapters,
)
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
        if model_has_adapters(model.model_path):
            logger.info(
                "HQQ quantization is not supported for models with adapters. Returning the model without quantization."
            )
            return model
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)
        ir_model = model.load_ir_model()
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1
        self._quantize_model(
            ir_model,
            config.nodes_to_exclude,
            config.nodes_to_include,
            config.block_size,
            config.axis,
            config.accuracy_level,
        )
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    def _quantize_model(
        self,
        ir_model: ir.Model,
        nodes_to_exclude: Optional[list[str]] = None,
        nodes_to_include: Optional[list[str]] = None,
        block_size: int = 128,
        axis: int = 0,
        accuracy_level: AccuracyLevel = AccuracyLevel.unset,
    ):
        nodes_to_exclude = nodes_to_exclude or []
        nodes_to_include = nodes_to_include or []

        ir_model.graph.sort()
        for node in ir_model.graph.all_nodes():
            node_name = node.name

            if node_name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node_name)
                continue

            elif node.op_type == OpType.MatMul and (node_name in nodes_to_include or not nodes_to_include):
                if not node.inputs[1].is_initializer():
                    logger.debug("skip to quantize %s as it has no initializer", node_name)
                    continue

                quantized_node, initializer_graph = self._quantize(node, block_size, axis, accuracy_level)

                if quantized_node.op_type == OpType.MatMulNBits:
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

    def _quantize(
        self, node: ir.Node, block_size: int, axis: int, accuracy_level: AccuracyLevel
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize the weight of the target node and return the new nodes.

        Target node:        QOperator node:
        MatMul              MatMulNBits
        If the node is target node with fp32 or fp16 const weight, quantize the weight to int4 and
        return the new nodes.
        return the corresponding QOperator nodes.
        """
        logger.debug("Start quantizing %s ...", node.name)

        if node.op_type == OpType.MatMul:
            return self._quantize_matmul(node, block_size, axis, accuracy_level)
        else:
            logger.error("Unsupported op %s for weight-only quantization.", node.op_type)
            return node, node.graph

    def _quantize_matmul(
        self, node: ir.Node, block_size: int, axis: int, accuracy_level: AccuracyLevel
    ) -> tuple[ir.Node, ir.Graph]:
        """Quantize weight B of MatMul node to int4.

        Currently only support 2D constant matrix and axis 0 blockwise quantization.
        """
        node_initializer = node.inputs[1]
        b_ndarray = node_initializer.const_value.numpy()

        if len(b_ndarray.shape) != 2:
            logger.debug("MatMul weight is not 2D. Skip to quantize")
            return node, node.graph  # can only process 2-D matrix

        packed, scales, zero_points = self._quantize_internal_numpy(b_ndarray, block_size, axis)

        b_quant = ir.Value(name=node_initializer.name + "_Q4", const_value=ir.tensor(packed))
        scales_tensor = ir.Value(name=node_initializer.name + "_scales", const_value=ir.tensor(scales))
        zero_points_tensor = ir.Value(name=node_initializer.name + "_zero_points", const_value=ir.tensor(zero_points))

        node_inputs = [node.inputs[0], b_quant, scales_tensor, zero_points_tensor]

        kwargs = {}
        rows, cols = b_ndarray.shape
        kwargs["K"] = rows
        kwargs["N"] = cols
        kwargs["bits"] = 4
        kwargs["block_size"] = block_size
        if accuracy_level > 0:
            kwargs["accuracy_level"] = accuracy_level

        node.outputs[0].name = node.outputs[0].name + "_Q4"

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.MatMulNBits),
            inputs=node_inputs,
            name=node.name + "_Q4" if node.name else "",
            attributes=kwargs,
        ), node_initializer.graph

    def _quantize_internal_numpy(self, b_ndarray, block_size: int, axis: int):
        """Convert numpy array to torch, quantize, and return numpy arrays."""
        b_array_torch = torch.from_numpy(b_ndarray)
        if torch.cuda.is_available():
            b_array_torch = b_array_torch.cuda()

        quant_weight_torch, scales_torch, zero_points_torch = self._quantize_internal(
            b_array_torch, group_size=block_size, axis=axis
        )
        quant_weight_torch = quant_weight_torch.contiguous()
        scales_torch = scales_torch.contiguous()
        zero_points_torch = zero_points_torch.contiguous()

        packed_torch = torch.zeros(
            (quant_weight_torch.shape[0], quant_weight_torch.shape[1] // 2),
            dtype=torch.uint8,
            device=quant_weight_torch.device,
        )
        self._pack_on_row_fast_248bit(packed_torch, quant_weight_torch)
        scales = scales_torch
        zero_points = zero_points_torch

        # reshape to the predefined shape in MatmulNbits
        scales = scales.reshape(-1)
        zero_points = zero_points.reshape(-1)
        rows, cols = b_array_torch.shape
        blob_size = block_size // 2
        k_blocks = (rows + block_size - 1) // block_size
        packed_torch = packed_torch.reshape(cols, k_blocks, blob_size)

        return packed_torch, scales, zero_points

    # Proximal solver || weight - dequantize(quantize(weight))||_p^p
    @staticmethod
    def _optimize_weights(tensor, scale, zero, min_max: list[int], axis: int = 0):
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
    def _pack_on_row_fast_248bit(pack_tensor, ori_int_tensor):
        if pack_tensor.shape[0] == ori_int_tensor.shape[0]:
            ori_int_tensor = ori_int_tensor.T
            pack_tensor = pack_tensor.T
        compress_ratio = pack_tensor.element_size() * 8 // 4  # 4 bits
        for j in range(compress_ratio):
            pack_tensor[0:] |= ori_int_tensor[j::compress_ratio] << (4 * (j))  # 4 bits

    # from Official implementation of Half-Quadratic Quantization (HQQ)
    def _quantize_internal(self, tensor, group_size: int = 128, axis: int = 0):
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
        weight = weight.reshape([-1, group_size]) if (axis == 1) else weight.reshape([group_size, -1])

        # Get min/max values
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
        zero = torch.round(-_min * scale)

        # Fine-tune weights
        scale, zero = self._optimize_weights(tensor=weight, scale=scale, zero=zero, min_max=min_max, axis=axis)

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
