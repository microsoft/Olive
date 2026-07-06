# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from pathlib import Path
from typing import Optional

import numpy as np
import numpy.typing as npt
import onnx_ir as ir

from olive.constants import MSFT_DOMAIN, OpType, Precision
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)

# 4-bit quantization types, must be consistent with the native ORT MatMulBnb4 kernel
# (Bnb_DataType_t defined in blockwise_quant_block_bnb4.h).
_BNB4_QUANT_TYPES = {
    # 4b floating point with bias of 3
    "fp4": 0,
    # 4b NormalFloat
    "nf4": 1,
}
_BNB4_BLOCK_SIZE = 64


class OnnxBnb4Quantization(Pass):
    """Quantize MatMul nodes in ONNX model using 4bit FP4/NF4 quantization."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "precision": PassConfigParam(
                type_=Optional[Precision],
                default_value=None,
                description="The quantization type. Only 'fp4' and 'nf4' are supported.",
            ),
            "quantized_modules": PassConfigParam(
                type_=list[str],
                description=(
                    "The list of modules to quantize. Node names will be matched as '.*[./]{module}[./]MatMul$'. If not"
                    " specified, all MatMul nodes will be quantized except those specified in nodes_to_exclude."
                ),
            ),
            "nodes_to_exclude": PassConfigParam(
                type_=list,
                default_value=None,
                description="List of node names to exclude from quantization.",
            ),
        }
        config.update(get_external_data_config())
        return config

    @classmethod
    def validate_config(
        cls,
        config: type[BasePassConfig],
        accelerator_spec: AcceleratorSpec,
    ) -> bool:
        if not super().validate_config(config, accelerator_spec):
            return False

        # Only fp4 and nf4 are supported
        return not config.precision or config.precision in {Precision.FP4, Precision.NF4}

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        precision = config.precision.value if config.precision else None
        quantized_modules = config.quantized_modules
        if model.model_attributes:
            quantized_modules = quantized_modules or model.model_attributes.get("quantized_modules")

            # extract precision from model_attributes if not specified in config
            if not precision:
                quant_config = model.model_attributes.get("quantization_config") or {}

                if quant_config.get("load_in_8bit"):
                    raise ValueError("load_in_8bit is not supported. Only 4-bit quantization is supported.")

                precision = quant_config.get("bnb_4bit_quant_type")
                if not precision:
                    raise ValueError("precision is required.")

                if quant_config.get("bnb_4bit_use_double_quant"):
                    logger.info(
                        "bnb_4bit_use_double_quant is set to True but double quantization is not supported. Ignoring."
                    )
        assert precision in _BNB4_QUANT_TYPES, f"quant_type must be one of 'fp4' or 'nf4'. Got {precision}."
        quant_type = _BNB4_QUANT_TYPES[precision]

        # load the model
        ir_model = model.load_ir_model()
        ir.external_data.load_to_model(ir_model)
        ir_model.graph.opset_imports[MSFT_DOMAIN] = 1

        # get nodes to exclude from quantization
        nodes_to_exclude = config.nodes_to_exclude or []

        # find all MatMul nodes in the graph
        matmul_nodes = self._find_matmul_nodes(ir_model.graph)
        # filter based on quantized_modules
        quantized_modules = set(quantized_modules or [])
        nodes_to_exclude = set(nodes_to_exclude) | {
            node
            for node in matmul_nodes
            if quantized_modules and not any(re.match(f".*[./]{key}[./]MatMul$", node) for key in quantized_modules)
        }

        # quantize the model
        self._quantize_model(ir_model, quant_type, nodes_to_exclude)

        # save the model to the output path and return the model
        return ir_model_to_olive_model(ir_model, output_model_path, config)

    def _quantize_model(self, ir_model: ir.Model, quant_type: int, nodes_to_exclude: set[str]) -> None:
        """Replace eligible MatMul nodes with MatMulBnb4 nodes carrying 4-bit quantized weights."""
        ir_model.graph.sort()
        for node in ir_model.graph.all_nodes():
            if node.op_type != str(OpType.MatMul):
                continue

            if node.name in nodes_to_exclude:
                logger.debug("exclude to quantize %s as specified by nodes_to_exclude...", node.name)
                continue

            quantized_node = self._quantize_matmul(node, quant_type)
            if quantized_node is node:
                # nothing changed (non-const or non-2D weight)
                continue

            weight_graph = node.inputs[1].graph
            for input_value in quantized_node.inputs:
                if input_value is not None and input_value.const_value is not None:
                    weight_graph.register_initializer(input_value)

            ir.convenience.replace_nodes_and_values(
                node.graph, node, [node], [quantized_node], node.outputs, quantized_node.outputs
            )

        self._remove_unused_initializers(ir_model)

    def _quantize_matmul(self, node: ir.Node, quant_type: int) -> ir.Node:
        """Quantize weight B of a MatMul node to 4-bit and return the new MatMulBnb4 node.

        Returns the original node unchanged if the weight is not a 2D constant.
        """
        logger.debug("start to quantize %s ...", node.name)

        node_initializer = node.inputs[1]
        if node_initializer is None or node_initializer.const_value is None:
            logger.debug("MatMul doesn't have const weight. Skip to quantize")
            return node

        b_ndarray = node_initializer.const_value.numpy()
        if len(b_ndarray.shape) != 2:
            logger.debug("MatMul weight is not 2D. Skip to quantize")
            return node

        packed, absmax = self._bnb4_block_quant(b_ndarray, quant_type)

        b_quant = ir.Value(name=node_initializer.name + "_Bnb4", const_value=ir.tensor(packed))
        absmax_value = ir.Value(name=node_initializer.name + "_absmax", const_value=ir.tensor(absmax))

        rows, cols = b_ndarray.shape
        kwargs = {
            "K": rows,
            "N": cols,
            "block_size": _BNB4_BLOCK_SIZE,
            "quant_type": quant_type,
        }

        self._rename_output_unless_graph_output(node)

        logger.debug("complete quantization of %s ...", node.name)

        return ir.node(
            domain=MSFT_DOMAIN,
            op_type=str(OpType.MatMulBnb4),
            inputs=[node.inputs[0], b_quant, absmax_value],
            name=node.name + "_Bnb4" if node.name else "",
            attributes=kwargs,
        )

    @staticmethod
    def _bnb4_block_quant(fpweight: npt.ArrayLike, quant_type: int) -> tuple[np.ndarray, np.ndarray]:
        """4b quantize fp32/fp16 weight using the native ORT bnb4 kernel."""
        from onnxruntime.capi._pybind_state import quantize_matmul_bnb4

        if len(fpweight.shape) != 2:
            raise ValueError("Current bnb4 block quantization only supports 2D tensors!")
        # need to copy since the transposed weight still has the original memory layout
        # Linear4bit quantizes its weight data which is the transposed weight
        fpweight_t = fpweight.transpose().copy()

        rows, cols = fpweight.shape
        numel = rows * cols
        block_size = _BNB4_BLOCK_SIZE
        num_blocks = (numel + block_size - 1) // block_size
        quantized_numel = (numel + 1) // 2

        packed = np.zeros(quantized_numel, dtype="uint8")
        absmax = np.zeros(num_blocks, dtype=fpweight.dtype)
        # block wise quantization, fpweight_t is flattened and divided into blocks
        quantize_matmul_bnb4(packed, fpweight_t, absmax, block_size, quant_type, cols, rows)

        return packed, absmax

    @staticmethod
    def _rename_output_unless_graph_output(node: ir.Node) -> None:
        """Append a `_Bnb4` suffix to the node's output name for renaming after quantization.

        Skips values that are graph outputs so external consumers relying on those names are
        not broken. Internal tensors are safe to rename because their consumers are rewired by
        replace_nodes_and_values.
        """
        graph = node.graph
        is_graph_output = graph is not None and node.outputs[0] in graph.outputs
        if not is_graph_output:
            node.outputs[0].name = node.outputs[0].name + "_Bnb4"

    @staticmethod
    def _remove_unused_initializers(ir_model: ir.Model) -> None:
        """Remove initializers that are no longer referenced by any node after quantization."""
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
            logger.debug("Removed %d unused initializers after quantization.", len(unused))

    @classmethod
    def _find_matmul_nodes(cls, graph: ir.Graph) -> list[str]:
        """Find all MatMul nodes in the graph (including subgraphs) and return their names."""
        return [node.name for node in graph.all_nodes() if node.op_type == str(OpType.MatMul)]
