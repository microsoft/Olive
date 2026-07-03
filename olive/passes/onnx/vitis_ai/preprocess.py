#
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#

from pathlib import Path

import onnx
import onnx_ir as ir
from onnx_ir.passes.common import TopologicalSortPass

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import model_proto_to_file, resave_model
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
        ir_model = ir.from_proto(model)
        graph = ir_model.graph

        input_names = set(config.input_names) if config.input_names else None
        inputs_to_update = [
            value
            for value in graph.inputs
            if value.shape is not None and len(value.shape) == 4 and (input_names is None or value.name in input_names)
        ]

        modified = False
        for value in inputs_to_update:
            old_dims = list(value.shape)
            input_rank = len(old_dims)

            # Move the channel dim (index 1) to the last position: NCHW -> NHWC
            new_dims = [old_dims[0], *old_dims[2:], old_dims[1]]

            # Transpose attribute to convert the NHWC input back to NCHW for consumers
            transpose_perm = list(range(input_rank))
            for i in range(input_rank):
                transpose_perm[i] = i if i < 1 else i - 1
            transpose_perm[1] = input_rank - 1

            # Capture consumers before inserting the transpose so we don't rewire the transpose itself
            consumers = [usage for usage in value.uses() if usage.node.op_type != "Transpose"]

            transpose_node = ir.Node(
                "",
                "Transpose",
                inputs=[value],
                attributes=[ir.AttrInt64s("perm", transpose_perm)],
                num_outputs=1,
                name=f"Transpose_{value.name}",
            )
            transpose_output = transpose_node.outputs[0]
            transpose_output.name = f"{value.name}_transpose"
            transpose_output.shape = ir.Shape(old_dims)
            transpose_output.type = value.type

            for usage in consumers:
                usage.node.replace_input_with(usage.idx, transpose_output)

            value.shape = ir.Shape(new_dims)
            graph.append(transpose_node)
            modified = True

        if modified:
            TopologicalSortPass()(ir_model)

        return ir.to_proto(ir_model)

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
