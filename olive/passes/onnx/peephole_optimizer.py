# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import copy
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
import onnxruntime as ort
from google.protobuf.message import Message
from onnx import TensorProto, helper, numpy_helper

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


# TODO(anyone): Move from onnxruntime.transformers.onnx_model.OnnxModel to OnnxDAG
# or reimplement logic using onnx-rewriter
# no need to create a new instance of OnnxModel for each optimization
class ModelOptimizer:
    def __init__(self, source_model_path):
        self.source_model_path = str(source_model_path)
        self.model = onnx.load(self.source_model_path)

    def remove_initializer_from_input(self) -> onnx.ModelProto:
        """Remove initializers from inputs in an ONNX model."""
        initializer_names = {initializer.name for initializer in self.model.graph.initializer}

        updated_inputs = [
            graph_input for graph_input in self.model.graph.input if graph_input.name not in initializer_names
        ]

        del self.model.graph.input[:]
        self.model.graph.input.extend(updated_inputs)

        return self.model

    def fuse_transpose_qat(self):
        # DequantizeLinear -> Transpose = Dequantize Linear with transposed initializer
        # Might need to check if this is performant for EPs like DML
        # Very limited use case, assumes a lot of things about the model
        # probably better to remove this or create a more general solution
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        onnx_model = TransformersOnnxModel(self.model)
        graph = onnx_model.graph()

        node_name2module = {}
        for node_idx, node in enumerate(graph.node):
            if node.name == "":
                node.name = str(node.op_type) + str(node_idx)
            node_name2module[node.name] = [node, node_idx]

        num_changed = 0
        for module in node_name2module.values():
            node = module[0]
            node_index = module[1]
            if node.op_type == "Transpose" and "DequantizeLinear" in node.input[0]:
                dequant_node_name = node.input[0][:-9]
                new_dequant_node_output = node.output[0]
                dequant_node = node_name2module[dequant_node_name][0]
                x_node = node_name2module[dequant_node.input[0][:-9]][0]
                x_scale_node = node_name2module[dequant_node.input[1][:-9]][0]
                x_zero_point_node = node_name2module[dequant_node.input[2][:-9]][0]

                x_val = onnx_model.get_constant_value(dequant_node.input[0])
                new_x_val = np.transpose(x_val, axes=(1, 0))
                x_scale_val = onnx_model.get_constant_value(dequant_node.input[1])
                x_zero_point_val = onnx_model.get_constant_value(dequant_node.input[2])

                self.remove_nodes(graph, [node, dequant_node, x_node, x_scale_node, x_zero_point_node])
                new_dequant, x, x_scale, x_zero_point = self.create_dequantizelinear_node(
                    new_x_val, x_scale_val, x_zero_point_val, new_dequant_node_output, node_index
                )
                self.add_nodes(graph, [new_dequant, x, x_scale, x_zero_point], node_index)
                num_changed += 1

        if num_changed > 0:
            logger.debug(
                "Converted %d Transpose -> DequantizeLinear to DequantizeLinear with transposed initializer",
                num_changed,
            )
            onnx_model.topological_sort()

    def create_dequantizelinear_node(self, x_val, x_scale_val, x_zero_point_val, outputs, node_name_suffix):
        x_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.INT8,
            dims=x_val.shape,
            vals=x_val.flatten().tobytes(),
            raw=True,
        )

        x_scale_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.FLOAT,
            dims=[1],
            vals=x_scale_val.tobytes(),
            raw=True,
        )

        x_zero_point_tensor = onnx.helper.make_tensor(
            name="const_tensor",
            data_type=onnx.TensorProto.DataType.INT8,
            dims=[1],
            vals=x_zero_point_val.tobytes(),
            raw=True,
        )

        x = onnx.helper.make_node("Constant", inputs=[], outputs=["x_" + str(node_name_suffix)], value=x_tensor)
        x_scale = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["x_scale_" + str(node_name_suffix)], value=x_scale_tensor
        )
        x_zero_point = onnx.helper.make_node(
            "Constant", inputs=[], outputs=["x_zero_point_" + str(node_name_suffix)], value=x_zero_point_tensor
        )

        dequant_node = onnx.helper.make_node(
            "DequantizeLinear",
            inputs=[
                "x_" + str(node_name_suffix),
                "x_scale_" + str(node_name_suffix),
                "x_zero_point_" + str(node_name_suffix),
            ],
            outputs=[outputs],
        )
        return dequant_node, x, x_scale, x_zero_point

    def remove_nodes(self, graph, nodes_list):
        for node in nodes_list:
            graph.node.remove(node)

    def add_nodes(self, graph, nodes_list, node_index):
        for node in nodes_list:
            graph.node.insert(node_index, node)

    @staticmethod
    def _create_node_name(nodes: Dict[str, Message], op_type: str, prefix_a: str, prefix_b: str):
        prefix: str = ""
        last_slash: int = -1
        for i in range(min(len(prefix_a), len(prefix_b))):
            if prefix_a[i] == prefix_b[i]:
                prefix += prefix_a[i]
                if prefix_a[i] == "/":
                    last_slash = i
            else:
                break

        if last_slash > 0:
            prefix = prefix[: last_slash + 1]

        if not prefix.endswith("/"):
            prefix += "/"
        prefix += op_type

        suffix: int = 0
        node_name: str = prefix
        while True:
            if node_name not in nodes:
                return node_name

            node_name = f"{prefix}_{suffix}"
            suffix += 1

    def patch_unsupported_argmax_operator(self):
        # ORT<1.20 doesn't support int64 input for ArgMax operator on CPU EP.
        # CUDA EP also falls back to CPU EP for non float inputs.
        # Add a cast node to convert int64 input to int32.
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        o_model = TransformersOnnxModel(self.model)
        o_model.topological_sort()  # Prerequisite for SymbolicShapeInference to succeed

        try:
            s_model = TransformersOnnxModel(
                SymbolicShapeInference.infer_shapes(o_model.model, auto_merge=True, guess_output_rank=True)
            )
        except Exception as e:
            logger.debug("Shape inference failed. Will try to continue without it. Error: %s", e)
            s_model = o_model

        o_nodes_by_name = {n.name: n for n in o_model.nodes()}
        s_value_info_by_name = {v.name: v for v in s_model.model.graph.value_info}
        s_value_info_by_name.update({n.name: n for n in s_model.model.graph.input})

        num_changed = 0
        for s_argmax_node in s_model.get_nodes_by_op_type("ArgMax"):
            if (s_value_info := s_value_info_by_name.get(s_argmax_node.input[0])) is None:
                # there is no value info for the input, so we can't determine the type
                continue
            if s_value_info.type.tensor_type.HasField("elem_type") and (
                s_value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64
            ):
                cast_node_name = ModelOptimizer._create_node_name(
                    o_nodes_by_name, "Cast", s_argmax_node.input[0], s_argmax_node.name
                )
                cast_output_name = cast_node_name + "_output_0"
                cast_node = onnx.helper.make_node(
                    "Cast", [s_argmax_node.input[0]], [cast_output_name], name=cast_node_name, to=onnx.TensorProto.INT32
                )

                o_argmax_node = o_nodes_by_name[s_argmax_node.name]
                o_argmax_node.input[0] = cast_output_name
                o_model.add_node(cast_node)
                o_nodes_by_name[cast_node_name] = cast_node

                num_changed += 1

        if num_changed > 0:
            logger.debug("Patched %d ArgMax operators with Cast operators", num_changed)
            o_model.topological_sort()

    def fuse_reshape_operations(self):
        # Remove unnecessary Reshape operator. Consecutive Reshape operators with latter's input being "[-1]"
        # i.e. flatten the input, the former Reshape operator is useless."""
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        o_model = TransformersOnnxModel(self.model)
        o_producers = o_model.output_name_to_node()
        o_consumers = o_model.input_name_to_nodes()

        expected_constant_value = np.array([-1])

        num_changed = 0
        for o_reshape_node in o_model.get_nodes_by_op_type("Reshape"):
            input_node_0 = o_producers.get(o_reshape_node.input[0])

            previous_is_reshape = input_node_0 and (input_node_0.op_type == "Reshape")
            current_flattens = o_model.get_constant_value(o_reshape_node.input[1]) == expected_constant_value
            only_consumer = len(o_consumers[o_reshape_node.input[0]]) == 1

            if previous_is_reshape and current_flattens and only_consumer:
                o_reshape_node.input[0] = input_node_0.input[0]
                input_node_0.input.remove(input_node_0.input[0])
                num_changed += 1

        if num_changed > 0:
            logger.debug("Fused %d redundant Reshape operators", num_changed)
            o_model.prune_graph()

    def _find_initializer_by_name(self, model, name):
        for initializer in model.graph.initializer:
            if initializer.name == name:
                return initializer
        raise ValueError(f"No initializer named {name}")

    def _find_value_info_proto_by_name(self, model, name):
        """Find the ValueInfoProto with the name name in the model's value_info."""
        for vi in model.graph.value_info:
            if vi.name == name:
                return vi

        for initializer in model.graph.initializer:
            if initializer.name == name:
                return helper.make_tensor_value_info(name, initializer.data_type, initializer.dims)

        for graph_input in model.graph.input:
            if graph_input.name == name:
                return graph_input

        raise ValueError(f"No value info proto named {name}")

    def _run_op(self, model, op):
        input_names = set()

        op_model = onnx.ModelProto()
        op_model.ir_version = model.ir_version
        op_model.producer_name = "constant_folding"
        op_model.opset_import.extend(model.opset_import)
        op_model.graph.name = "ConstantFoldingGraph"
        op_model.graph.node.extend([copy.deepcopy(op)])

        for input_name in op.input:
            if input_name and input_name not in input_names:
                try:
                    initializer = self._find_initializer_by_name(model, input_name)
                    op_model.graph.initializer.append(copy.deepcopy(initializer))
                    vi = helper.make_tensor_value_info(initializer.name, initializer.data_type, initializer.dims)
                    op_model.graph.input.append(vi)
                except ValueError:
                    vi = self._find_value_info_proto_by_name(model, input_name)
                    op_model.graph.input.append(copy.deepcopy(vi))
                input_names.add(input_name)

        for output_name in op.output:
            vi = helper.make_tensor_value_info(output_name, TensorProto.UNDEFINED, [])
            op_model.graph.output.append(vi)

        return op_model

    def _run_onnx_model(self, model):
        session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
        input_dict = {}
        for model_input in session.get_inputs():
            name = model_input.name
            tensor = self._find_initializer_by_name(model, name)
            input_dict[name] = numpy_helper.to_array(tensor)
        return session.run(None, input_dict)

    def _get_constant_nodes(self, model):
        dynamic_inputs = {graph_input.name for graph_input in model.graph.input}
        const_inputs = {
            initializer.name for initializer in model.graph.initializer if initializer.name not in dynamic_inputs
        }
        const_nodes = []
        for node in model.graph.node:
            if all(node_input == "" or node_input in const_inputs for node_input in node.input):
                const_nodes.append(node)
                const_inputs.update(node.output)
        return const_nodes

    def fold_constant(self):
        model_copy = copy.deepcopy(self.model)

        while True:
            const_nodes = self._get_constant_nodes(model_copy)
            if not const_nodes:
                break

            nodes_to_remove = []
            for node in const_nodes:
                try:
                    op_model = self._run_op(model_copy, node)
                    outputs = self._run_onnx_model(op_model)
                    for output_array, name in zip(outputs, node.output):
                        if any(init.name == name for init in model_copy.graph.initializer):
                            continue
                        tensor = numpy_helper.from_array(output_array, name)
                        model_copy.graph.initializer.append(tensor)
                        vi = helper.make_tensor_value_info(name, tensor.data_type, tensor.dims)
                        model_copy.graph.value_info.append(vi)
                    nodes_to_remove.append(node)
                except Exception as e:
                    logger.warning("Failed to run %s op (name is %s): %s, skip...", node.op_type, node.name, e)

            for node in nodes_to_remove:
                model_copy.graph.node.remove(node)
        self.model = model_copy


class OnnxPeepholeOptimizer(Pass):
    """Optimize ONNX model by fusing nodes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # optimize model
        peephole_optimizer = ModelOptimizer(model.model_path)
        peephole_optimizer.fuse_transpose_qat()
        peephole_optimizer.patch_unsupported_argmax_operator()
        peephole_optimizer.fuse_reshape_operations()
        peephole_optimizer.fold_constant()
        peephole_optimizer.remove_initializer_from_input()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(peephole_optimizer.model, output_model_path, config)
