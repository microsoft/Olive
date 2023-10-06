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
from onnx.mapping import NP_TYPE_TO_TENSOR_TYPE
from onnx.onnx_pb import GraphProto, NodeProto, TensorProto

from olive.hardware import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class QuantType(Enum):
    General8bit = 0
    FP4 = 1
    NF4 = 2


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

        # has_ms_domain = False
        # for opset in opset_import:
        #     if opset.domain == "com.microsoft":
        #         has_ms_domain = True
        # if not has_ms_domain:
        #     opset_import.extend([onnx.helper.make_opsetid("com.microsoft", 1)])
        opset_import.extend([onnx.helper.make_opsetid("olive", 1)])

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

            new_nodes.extend(cls.create_matmul_bnb4_nodes(node, graph_stack, quantized_modules, quantization_config))

        graph.ClearField("node")
        graph.node.extend(new_nodes)
        graph_stack.pop()
        return graph

    @classmethod
    def create_matmul_bnb4_nodes(
        cls,
        node: NodeProto,
        graph_stack: List[GraphProto],
        quantized_modules: List[str],
        quantization_config: Dict[str, Any],
    ) -> NodeProto:
        """Create a BnbDequantize node and a MatMul node for the given MatMul node.

        If the node is MatMul with const weight and part of quantized_modules, quantize the weight with 4bit.
        Create a BnbDequantize node and a MatMul node to replace the original MatMul node.
        """
        if node.op_type != "MatMul":
            return [node]  # only care about MatMul for now

        is_quantized_module = False
        for module in quantized_modules:
            if node.name.endswith(f"{module}/MatMul"):
                is_quantized_module = True
                break
        if not is_quantized_module:
            return [node]

        inputB = node.input[1]  # noqa: N806
        B, Bs_graph = cls.get_initializer(inputB, graph_stack)  # noqa: N806
        if B is None:
            return [node]  # only care about constant weight

        B_array = onnx.numpy_helper.to_array(B)  # noqa: N806
        if len(B_array.shape) != 2:
            return [node]  # can only process 2-D matrix

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
        absmax, shape, dtype, blocksize, compressed_stats, quant_type, data_type = quant_state
        del data_type

        B_absmax = onnx.numpy_helper.from_array(absmax.cpu().numpy())  # noqa: N806
        B_absmax.name = B.name + "_absmax"

        B_shape = onnx.numpy_helper.from_array(np.array([shape[0], shape[1]]).astype(np.int64))  # noqa: N806
        B_shape.name = B.name + "_shape"

        Bs_graph.initializer.extend([B_quant, B_absmax, B_shape])

        kwargs = {}
        # TODO(jambayk): Generalize this to support other dtypes, create a utility function
        torch_dtype_to_onnx_dtype = {
            torch.float32: TensorProto.FLOAT,
            torch.float16: TensorProto.FLOAT16,
            torch.bfloat16: TensorProto.BFLOAT16,
        }
        kwargs["dtype"] = torch_dtype_to_onnx_dtype[dtype]
        kwargs["blocksize"] = blocksize
        # only need to worry about nf4 and fp4 for now
        kwargs["quant_type"] = QuantType[quant_type.upper()].value
        kwargs["double_quantized"] = compressed_stats is not None

        nested_inputs = []
        if compressed_stats:
            offset, quant_state2 = compressed_stats
            # nested_absmax is a tensor, torch.float32 -> initializer
            # nested_code is a tensor, torch.float32 -> initializer
            # nested_blacksize is an int -> attribute
            # the rest are always False, torch.float32, None, None
            nested_absmax, nested_code, nested_blocksize, _, _, _, _ = quant_state2

            B_offset = onnx.numpy_helper.from_array(offset.cpu().numpy())  # noqa: N806
            B_offset.name = B.name + "_offset"

            B_nested_absmax = onnx.numpy_helper.from_array(nested_absmax.cpu().numpy())  # noqa: N806
            B_nested_absmax.name = B.name + "_nested_absmax"

            # TODO(jambayk): nested_code is same for all nodes, can we make it shared?
            B_nested_code = onnx.numpy_helper.from_array(nested_code.cpu().numpy())  # noqa: N806
            B_nested_code.name = B.name + "_nested_code"

            Bs_graph.initializer.extend([B_offset, B_nested_absmax, B_nested_code])

            # create nested inputs and attributes
            nested_inputs = [B_offset.name, B_nested_absmax.name, B_nested_code.name]
            kwargs["nested_blocksize"] = nested_blocksize

        dequantize_node_name = node.name + "_BnbDequantize"
        dequantize_node_output = onnx.helper.make_tensor_value_info(
            name=dequantize_node_name + "_output",
            elem_type=NP_TYPE_TO_TENSOR_TYPE[B_array.dtype],
            shape=[B_array.shape[1], B_array.shape[0]],
        )
        Bs_graph.value_info.append(dequantize_node_output)
        dequantize_node = onnx.helper.make_node(
            "BnbDequantize",
            inputs=[node.input[0], B_quant.name, B_absmax.name, B_shape.name, *nested_inputs],
            outputs=[dequantize_node_output.name],
            name=dequantize_node_name,
            domain="olive",
            **kwargs,
        )
        transpose_node = onnx.helper.make_node(
            "Transpose",
            inputs=[dequantize_node_output.name],
            outputs=[dequantize_node_output.name + "_transposed"],
            name=dequantize_node_name + "_transpose",
            perm=[1, 0],
        )
        matmul_node = onnx.helper.make_node(
            "MatMul",
            inputs=[node.input[0], dequantize_node_output.name + "_transposed"],
            outputs=[node.output[0]],
            name=node.name,
            **{attr.name: attr for attr in node.attribute},
        )
        return [dequantize_node, transpose_node, matmul_node]

    @staticmethod
    @torch.no_grad()
    def quantize_weight(weight: np.ndarray, quantization_config: Dict[str, Any]) -> Tuple[np.ndarray, Any]:
        """Quantize weight using bitsandbytes."""
        # from bitsandbytes.functional import quantize_4bit
        import bitsandbytes as bnb

        # NOTE: bitsandbytes Linear4bit always uses float16 as backend dtype
        # not sure if this is intentional since they have kernels for float32 and bfloat16
        # https://github.com/TimDettmers/bitsandbytes/blob/0.41.0/bitsandbytes/nn/modules.py#L156
        # we use float16 during quantization (through the Linear4bit module) but use the original dtype
        # during dequantization
        # otherwise, there are complications with the typing of the dequantization op
        # TODO(jambayk): look into this again if model accuracy suffers for float models
        # for some reason directly calling quantize_4bit doesn't give the same result as using the Linear4bit module
        # .copy() to avoid numpy not writable warning when converting to torch
        weight = torch.from_numpy(weight.copy()).t()
        bnb_linear = bnb.nn.Linear4bit(
            weight.shape[1],
            weight.shape[0],
            bias=False,
            compress_statistics=quantization_config["bnb_4bit_use_double_quant"],
            quant_type=quantization_config["bnb_4bit_quant_type"],
        )
        bnb_linear.weight.data = weight
        bnb_linear.cuda()
        return bnb_linear.weight.data.t(), bnb_linear.weight.quant_state

    @staticmethod
    def get_initializer(name, graph_path: List[GraphProto]) -> Tuple[TensorProto, GraphProto]:
        for gid in range(len(graph_path) - 1, -1, -1):
            graph = graph_path[gid]
            for tensor in graph.initializer:
                if tensor.name == name:
                    return tensor, graph
        return None, None
