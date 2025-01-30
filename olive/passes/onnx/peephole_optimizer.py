# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
import onnx
from google.protobuf.message import Message

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

    def clip_attention_mask(self):
        num_changed = 0
        for n in self.model.graph.node:
            if n.op_type == "ConstantOfShape":
                value = onnx.helper.get_attribute_value(n.attribute[0])
                tensor_value = onnx.numpy_helper.to_array(value)
                if tensor_value[0] < -3e30:
                    new_attribute = onnx.helper.make_attribute(
                        "value", onnx.helper.make_tensor(value.name, value.data_type, [1], [-3e30])
                    )
                    n.ClearField("attribute")
                    n.attribute.extend([new_attribute])
                    num_changed += 1

        if num_changed > 0:
            logger.debug("Replaced %d attention mask values with -3e30", num_changed)

    def remove_useless_cast_nodes(self):
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        o_model = TransformersOnnxModel(self.model)
        o_model.remove_useless_cast_nodes()

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
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        # optimize model
        peephole_optimizer = ModelOptimizer(model.model_path)
        peephole_optimizer.onnxscript_optimize()
        peephole_optimizer.onnxoptimizer_optimize()
        peephole_optimizer.clip_attention_mask()
        peephole_optimizer.remove_useless_cast_nodes()
        peephole_optimizer.patch_unsupported_argmax_operator()
        peephole_optimizer.fuse_reshape_operations()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(peephole_optimizer.model, output_model_path, config)
