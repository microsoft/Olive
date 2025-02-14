# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Dict, Type

import numpy as np
import onnx
from google.protobuf.message import Message

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


# TODO(anyone): Move from onnxruntime.transformers.onnx_model.OnnxModel to OnnxDAG
# or reimplement logic using onnx-rewriter
# no need to create a new instance of OnnxModel for each optimization
class ModelOptimizer:
    def __init__(self, source_model_path):
        self.source_model_path = str(source_model_path)
        self.model = onnx.load(self.source_model_path)

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

    def onnxscript_optimize(self):
        try:
            import onnxscript
        except ImportError:
            logger.warning("Please install `onnxscript` to apply more optimization.")
            return

        onnxscript.optimizer.optimize(self.model)

    def onnxoptimizer_optimize(self):
        try:
            from onnxoptimizer import optimize
        except ImportError:
            logger.warning("Please install `onnxoptimizer` to apply more optimization.")
            return

        self.model = optimize(self.model)


class OnnxPeepholeOptimizer(Pass):
    """Optimize ONNX model by fusing nodes."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # optimize model
        peephole_optimizer = ModelOptimizer(model.model_path)
        peephole_optimizer.onnxscript_optimize()
        peephole_optimizer.onnxoptimizer_optimize()
        peephole_optimizer.fuse_transpose_qat()
        peephole_optimizer.patch_unsupported_argmax_operator()
        peephole_optimizer.fuse_reshape_operations()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(peephole_optimizer.model, output_model_path, config)
