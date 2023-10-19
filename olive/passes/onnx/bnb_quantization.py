# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from enum import Enum
from typing import Any, Dict, List, Tuple

import numpy as np
import onnx
import torch
from onnx.onnx_pb import GraphProto, NodeProto, TensorProto

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class QuantType(Enum):
    FP4 = 0
    NF4 = 1


class OnnxBNBQuantization(Pass):
    """Quantize MatMul nodes in ONNX model using BitsAndBytes quantization.

    Only supports 4-bit quantization for now.
    """

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "quantization_config": PassConfigParam(
                type_=Dict[str, Any],
                description=(
                    "The quantization config for BitsAndBytes quantization. Expected to be the same schema as"
                    " transformers.BitsAndBytesConfig.to_dict()."
                ),
            ),
            "quantized_modules": PassConfigParam(
                type_=List[str],
                description=(
                    "The list of modules to quantize. Will match the end of MatMul nodes with this list to determine"
                    " which nodes to quantize."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        quantization_config = config["quantization_config"]
        quantized_modules = config["quantized_modules"]
        if model.model_attributes:
            model_attributes = model.model_attributes
            quantization_config = quantization_config or model_attributes.get("quantization_config")
            quantized_modules = quantized_modules or model_attributes.get("quantized_modules")
        if not quantization_config:
            raise ValueError("quantization_config is required.")
        if not quantized_modules:
            raise ValueError("quantized_modules is required.")

        if quantization_config.get("load_in_8bit"):
            raise ValueError("load_in_8bit is not supported. Only 4-bit quantization is supported.")

        if quantization_config.get("bnb_4bit_use_double_quant"):
            logger.info("bnb_4bit_use_double_quant is set to True but double quantization is not supported. Ignoring.")

        for key in ["bnb_4bit_quant_type"]:
            if key not in quantization_config:
                raise ValueError(f"{key} is required in quantization_config.")

        output_model_path = ONNXModel.resolve_path(output_model_path)

        onnx_model = model.load_model()
        # use a stack to keep track of sub-graphs
        graph_stack = [onnx_model.graph]
        # prepare for quantization by adding the common quantization initializers
        quantization_info = {
            "config": quantization_config,
            "modules": quantized_modules,
        }

        # add the olive opset
        opset_import = onnx_model.opset_import
        has_ms_domain = False
        for opset in opset_import:
            if opset.domain == "com.microsoft":
                has_ms_domain = True
        if not has_ms_domain:
            opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

        self.process_subgraph(graph_stack, quantization_info)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    @classmethod
    def process_subgraph(cls, graph_stack: List[GraphProto], quantization_info: Dict[str, Any]) -> GraphProto:
        from onnxruntime.quantization.quant_utils import attribute_to_kwarg

        new_nodes = []
        graph = graph_stack[-1]

        for node in graph.node:
            graph_attrs = [
                attr
                for attr in node.attribute
                if attr.type == onnx.AttributeProto.GRAPH or attr.type == onnx.AttributeProto.GRAPHS
            ]
            if len(graph_attrs):
                kwargs = {}
                for attr in node.attribute:
                    if attr.type == onnx.AttributeProto.GRAPH:
                        # recursive call to take care of sub-graph
                        graph_stack.append(attr.g)
                        kv = {attr.name: cls.process_subgraph(graph_stack, quantization_info)}
                    elif attr.type == onnx.AttributeProto.GRAPHS:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([cls.process_subgraph(graph_stack, quantization_info)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            new_nodes.append(cls.create_matmul_bnb4_node(node, graph_stack, quantization_info))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    @classmethod
    def create_matmul_bnb4_node(
        cls, node: NodeProto, graph_stack: List[GraphProto], quantization_info: Dict[str, Any]
    ) -> NodeProto:
        """Create a MatMulBnb4 node from a MatMul node.

        If the node is Matmul with const 2D weight and part of quantized_modules, quantize the weight with 4bit.
        Create a MatMulBnb4 node to replace the original MatMul node.
        """
        # only care about Matmul for now
        if node.op_type != "MatMul":
            return node

        is_quantized_module = False
        for module in quantization_info["modules"]:
            if node.name.endswith(f"{module}/MatMul"):
                is_quantized_module = True
                break
        if not is_quantized_module:
            return node

        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = cls.get_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            return node  # only care about constant weight

        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            return node  # can only process 2-D matrix

        # weight_4bit, uint8 -> initializer
        # absmax, same type as weight -> initializer
        # block_size, int -> attribute
        # quant_type, str -> enum -> attribute
        weight_4bit, absmax, block_size, quant_type_enum = cls.quantize_weight(B_array, quantization_info["config"])

        B_quant = onnx.numpy_helper.from_array(weight_4bit)  # noqa: N806
        B_quant.name = B.name + "_w4bit"
        Bs_graph.initializer.remove(B)
        for graph_input in Bs_graph.input:
            if graph_input.name == inputB:
                Bs_graph.input.remove(graph_input)
                break

        B_absmax = onnx.numpy_helper.from_array(absmax)  # noqa: N806
        B_absmax.name = B.name + "_absmax"

        Bs_graph.initializer.extend([B_quant, B_absmax])

        kwargs = {}
        rows, cols = B_array.shape
        kwargs["K"] = rows  # in_features
        kwargs["N"] = cols  # out_features
        kwargs["block_size"] = block_size
        kwargs["quant_type"] = quant_type_enum

        return onnx.helper.make_node(
            "MatMulBnb4",
            inputs=[node.input[0], B_quant.name, B_absmax.name],
            outputs=[node.output[0]],
            name=node.name + "_Bnb4",
            domain="com.microsoft",
            **kwargs,
        )

    @staticmethod
    @torch.no_grad()
    def quantize_weight(weight: np.ndarray, quantization_config: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """Quantize weight using bitsandbytes."""
        from onnxruntime.capi._pybind_state import quantize_matmul_bnb4

        quant_type = quantization_config["bnb_4bit_quant_type"]
        # only need to worry about nf4 and fp4
        quant_type_enum = QuantType[quant_type.upper()].value

        # need to copy since the transposed weight still has the original memory layout
        # Linear4bit quantizes its weight data which is the transposed weight
        weight_t = weight.transpose().copy()

        K, N = weight.shape  # noqa: N806
        numel = N * K
        block_size = 64  # bnb.nn.Linear4bit always uses block_size=64
        num_blocks = (numel + block_size - 1) // block_size
        quantized_numel = (numel + 1) // 2

        weight_4bit = np.zeros(quantized_numel, dtype=np.uint8)
        absmax = np.zeros(num_blocks, dtype=weight.dtype)
        quantize_matmul_bnb4(weight_4bit, weight_t, absmax, block_size, quant_type_enum, N, K)

        return weight_4bit, absmax, block_size, quant_type_enum

    @staticmethod
    def get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None
