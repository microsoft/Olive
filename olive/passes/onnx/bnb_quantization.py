# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
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
            raise ValueError("load_in_8bit is not supported for now.")

        for key in ["bnb_4bit_use_double_quant", "bnb_4bit_quant_type"]:
            if key not in quantization_config:
                raise ValueError(f"{key} is required in quantization_config.")

        output_model_path = ONNXModel.resolve_path(output_model_path)

        onnx_model = model.load_model()
        # use a stack to keep track of sub-graphs
        graph_stack = [onnx_model.graph]
        opset_import = onnx_model.opset_import

        has_ms_domain = False
        for opset in opset_import:
            if opset.domain == "com.microsoft":
                has_ms_domain = True
        if not has_ms_domain:
            opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])

        self.process_subgraph(graph_stack, quantized_modules, quantization_config)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model, output_model_path, config)

    @classmethod
    def process_subgraph(
        cls, graph_stack: List[GraphProto], quantized_modules: List[str], quantization_config: Dict[str, Any]
    ) -> GraphProto:
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
                        kv = {attr.name: cls.process_subgraph(graph_stack, quantized_modules, quantization_config)}
                    elif attr.type == onnx.AttributeProto.GRAPH:
                        value = []
                        for subgraph in attr.graphs:
                            # recursive call to take care of sub-graph
                            graph_stack.append(subgraph)
                            value.extend([cls.process_subgraph(graph_stack, quantized_modules, quantization_config)])
                        kv = {attr.name: value}
                    else:
                        kv = attribute_to_kwarg(attr)
                    kwargs.update(kv)
                node = onnx.helper.make_node(  # noqa: PLW2901
                    node.op_type, node.input, node.output, name=node.name, **kwargs
                )

            new_nodes.append(cls.create_matmul_bnb4_node(node, graph_stack, quantized_modules, quantization_config))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    @classmethod
    def create_matmul_bnb4_node(
        cls,
        node: NodeProto,
        graph_stack: List[GraphProto],
        quantized_modules: List[str],
        quantization_config: Dict[str, Any],
    ) -> NodeProto:
        """Create a MatMulBnb4 node from a MatMul node.

        If the node is MatMul with const weight and part of quantized_modules, quantize the weight with 4bit, and
        return the new node.
        """
        if node.op_type != "MatMul":
            return node  # only care about MatMul for now

        is_quantized_module = False
        for module in quantized_modules:
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

        # torch.uint8 -> initializer
        weight_4bit, quant_state = cls.quantize_weight(B_array, quantization_config)

        B_quant = onnx.numpy_helper.from_array(weight_4bit.cpu().numpy())  # noqa: N806
        B_quant.name = B.name + "_w4bit"
        Bs_graph.initializer.remove(B)
        for graph_input in Bs_graph.input:
            if graph_input.name == inputB:
                Bs_graph.input.remove(graph_input)
                break

        # absmax is tensor, torch.uint8 -> initializer (not sure if it is always an array of 1 element)
        # shape is torch.Size, 2 elements -> attribute or initializer?
        # dtype is torch.dtype -> attribute (currently, it is always torch.float16)
        # blocksize is int -> attribute
        # quant_type is str -> attribute
        # data_type is an array mapping from quantized value to original value
        # but we don't need to use data_type since it's already hard-coded in the kernel
        # compressed_stats is a another quantization stat for the double quantization
        # TODO(jambayk): handle compressed_stats
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, data_type = quant_state
        del data_type

        B_absmax = onnx.numpy_helper.from_array(absmax.cpu().numpy())  # noqa: N806
        B_absmax.name = B.name + "_absmax"

        B_shape = onnx.numpy_helper.from_array(np.array([shape[0], shape[1]]).astype(np.int64))  # noqa: N806
        B_shape.name = B.name + "_shape"

        Bs_graph.initializer.extend([B_quant, B_shape])

        kwargs = {}
        # TODO(jambayk): find an appropriate attribute type for dtype
        kwargs["dtype"] = str(dtype)
        kwargs["blocksize"] = blocksize
        kwargs["quant_type"] = quant_type
        return onnx.helper.make_node(
            "MatMulBnb4",
            inputs=[node.input[0], B_quant.name, B_absmax.name, B_shape.name],
            outputs=[node.output[0]],
            name=node.name + "_Bnb4" if node.name else "",
            domain="com.microsoft",
            **kwargs,
        )

    @staticmethod
    def quantize_weight(weight: np.ndarray, quantization_config: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """Quantize weight using bitsandbytes."""
        from bitsandbytes.functional import quantize_4bit

        # NOTE: bitsandbytes Linear4bit always uses float16 as backend dtype
        # not sure if this is intentional since they have kernels for float32 and bfloat16
        # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L156
        # we will use float16 for now
        weight = torch.from_numpy(weight).half().cuda()
        weight_4bit, quant_state = quantize_4bit(
            weight,
            compress_statistics=quantization_config["bnb_4bit_use_double_quant"],
            quant_type=quantization_config["bnb_4bit_quant_type"],
        )
        weight_4bit = weight_4bit.t()
        return weight_4bit, quant_state

    @staticmethod
    def get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None
