#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
from olive.passes.onnx.onnx_dag import OnnxDAG
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class InputNCHWtoNHWC(Pass):
    """Updates model inputs from NCHW to NHWC by adding a transpose node at the input."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "input_names": PassConfigParam(
                type_=list,
                required=False,
                description="Model inputs names to transpose",
                default_value=None,
            )
        }

    def _add_input_transpose(self, model: onnx.ModelProto, config: BasePassConfig) -> onnx.ModelProto:
        """Add transpose node at model's NCHW inputs."""
        dag = OnnxDAG(model)
        graph = dag.model.graph
        orig_inval_info = {tensor.name: tensor for tensor in graph.input}
        inputs_to_update = None
        if config.input_names:
            inputs_to_update = [
                tensor.name
                for tensor in graph.input
                if tensor.name in config.input_names and len(tensor.type.tensor_type.shape.dim) == 4
            ]
        else:
            inputs_to_update = [tensor.name for tensor in graph.input if len(tensor.type.tensor_type.shape.dim) == 4]

        transpose_output_names = {}
        transpose_nodes = []
        for iname in inputs_to_update:
            tensor = orig_inval_info[iname]
            input_rank = len(tensor.type.tensor_type.shape.dim)
            channel_dim = onnx.TensorShapeProto.Dimension()
            ishape = tensor.type.tensor_type.shape
            channel_dim.CopyFrom(ishape.dim[1])
            for i in range(1, input_rank - 1):
                ishape.dim[i].CopyFrom(ishape.dim[i + 1])
            ishape.dim[input_rank - 1].CopyFrom(channel_dim)

            # Transpose attribute
            transpose_perm = list(range(input_rank))
            for i in range(input_rank):
                transpose_perm[i] = i if i < 1 else i - 1
            transpose_perm[1] = input_rank - 1

            transpose_output_names[iname] = iname + "_transpose"
            transpose_node = onnx.helper.make_node(
                "Transpose",
                name=f"Transpose_{iname}",
                inputs=[iname],
                outputs=[transpose_output_names[iname]],
                perm=transpose_perm,
            )
            # Add to graph nodes
            transpose_nodes.append(transpose_node)
            dag.add_node(transpose_node, 0)

        for name in inputs_to_update:
            consumers = dag.get_consumers(name)
            for cnode in consumers:
                if dag.get_node_op_type(cnode) != "Transpose":
                    dag.replace_node_input(cnode, name, transpose_output_names[name])

        dag.update()
        return dag.model

    def _run_for_config(
        self, model: ONNXModelHandler, config: BasePassConfig, output_model_path: str
    ) -> ONNXModelHandler:
        if not isinstance(model, ONNXModelHandler):
            raise ValueError("Model must be an instance of ONNXModelHandler")

        output_model_path = Path(resolve_onnx_path(output_model_path, Path(model.model_path).name))

        # resave the original model to the new path
        has_external_data = resave_model(model.model_path, output_model_path)
        # load the model without external data
        onnx_model = self._add_input_transpose(onnx.load_model(output_model_path, load_external_data=False), config)
        # save the model with metadata, will unlink to avoid modifying the hardlinked original
        model_proto_to_file(onnx_model, output_model_path)

        return ONNXModelHandler(
            model_path=output_model_path.parent if has_external_data else output_model_path,
            onnx_file_name=output_model_path.name if has_external_data else None,
        )
