# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from typing import Any, Dict

from google.protobuf.message import Message

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OrtPatchArgMaxOperator(Pass):
    """Insert a Cast operator ahead of ArgMax if input tensor type is of unsupported type."""

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {}
        config.update(get_external_data_config())
        return config

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

    def _run_for_config(
        self, model: ONNXModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModel:
        import onnx
        from onnxruntime.tools.symbolic_shape_infer import SymbolicShapeInference
        from onnxruntime.transformers.onnx_model import OnnxModel

        o_model = OnnxModel(model.load_model())
        o_model.topological_sort()

        s_model = OnnxModel(
            SymbolicShapeInference.infer_shapes(
                o_model.model,
                auto_merge=True,
                guess_output_rank=True,
                # verbose=1,
            )
        )

        o_nodes_by_name = {n.name: n for n in o_model.nodes()}
        s_value_info_by_name = {v.name: v for v in s_model.model.graph.value_info}
        s_value_info_by_name.update({n.name: n for n in s_model.model.graph.input})

        for s_argmax_node in s_model.get_nodes_by_op_type("ArgMax"):
            s_value_info = s_value_info_by_name[s_argmax_node.input[0]]
            if s_value_info.type.tensor_type.HasField("elem_type") and (
                s_value_info.type.tensor_type.elem_type == onnx.TensorProto.INT64
            ):

                cast_node_name = OrtPatchArgMaxOperator._create_node_name(
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

        return model_proto_to_olive_model(o_model.model, output_model_path, config)
