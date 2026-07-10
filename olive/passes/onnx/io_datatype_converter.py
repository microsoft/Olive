# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from pathlib import Path
from typing import Optional

import onnx
import onnx_ir as ir

from olive.constants import OpType
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxIODataTypeConverter(Pass):
    """Converts model inputs/outputs from a source dtype to a target dtype based on a name pattern."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "name_pattern": PassConfigParam(
                type_=str,
                default_value="logits",
                description=(
                    "Only convert inputs/outputs whose name matches this pattern. By defaultlooking for logits names"
                ),
            ),
            "source_dtype": PassConfigParam(
                type_=int,
                default_value=10,
                description="Source data type int value to convert from (default: FLOAT16). Check "
                "https://github.com/onnx/onnx/blob/96a0ca4374d2198944ff882bd273e64222b59cb9/onnx/onnx.proto3#L503-L551"
                "for details.",
            ),
            "target_dtype": PassConfigParam(
                type_=int,
                default_value=1,
                description="Target data type int value to convert to (default: FLOAT). Check "
                "https://github.com/onnx/onnx/blob/96a0ca4374d2198944ff882bd273e64222b59cb9/onnx/onnx.proto3#L503-L551"
                "for details.",
            ),
        }
        config.update(get_external_data_config())
        return config

    def _wrap_inputs(
        self, graph: ir.Graph, names: Optional[re.Pattern], source_dtype: ir.DataType, target_dtype: ir.DataType
    ) -> int:
        # 1. find source_dtype graph inputs
        # 2. rewrite all consumers to read from a Cast output
        # 3. insert Cast that converts the (now target_dtype) input back to source_dtype
        # 4. rewrite the graph input dtype to target_dtype
        converted_count = 0
        for graph_input in list(graph.inputs):
            if graph_input.dtype != source_dtype:
                continue
            if not self._is_name_matched(graph_input.name, names):
                continue
            logger.debug("Converting input %s from %s to %s", graph_input.name, source_dtype, target_dtype)

            cast_out = ir.Value(
                name=graph_input.name + "_converted", shape=graph_input.shape, type=ir.TensorType(source_dtype)
            )
            # redirect existing consumers before the Cast node exists so it is not itself redirected
            graph_input.replace_all_uses_with(cast_out)

            cast_node = ir.node(
                str(OpType.Cast), inputs=[graph_input], attributes={"to": int(source_dtype)}, outputs=[cast_out]
            )
            graph_input.type = ir.TensorType(target_dtype)
            graph.append(cast_node)
            converted_count += 1

        return converted_count

    def _wrap_outputs(
        self, graph: ir.Graph, names: Optional[re.Pattern], source_dtype: ir.DataType, target_dtype: ir.DataType
    ) -> int:
        # 1. find source_dtype graph outputs
        # 2. rename the internal tensor (keeping its source_dtype consumers intact)
        # 3. append a Cast that converts the internal tensor to target_dtype under the original output name
        # 4. rewrite the graph output slot to the Cast output
        converted_count = 0
        for idx, graph_output in list(enumerate(graph.outputs)):
            if graph_output is None or graph_output.dtype != source_dtype:
                continue
            if not self._is_name_matched(graph_output.name, names):
                continue
            logger.debug("Converting output %s from %s to %s", graph_output.name, source_dtype, target_dtype)

            original_name = graph_output.name
            # keep the internal tensor in source_dtype; all its existing consumers follow the rename
            graph_output.name = original_name + "_converted"

            cast_node = ir.node(str(OpType.Cast), inputs=[graph_output], attributes={"to": int(target_dtype)})
            cast_out = cast_node.outputs[0]
            cast_out.name = original_name
            cast_out.shape = graph_output.shape
            cast_out.type = ir.TensorType(target_dtype)

            graph.outputs[idx] = cast_out
            graph.append(cast_node)
            converted_count += 1

        return converted_count

    def _is_name_matched(self, name: str, names: Optional[re.Pattern]) -> bool:
        return not names or bool(names.search(name))

    def _get_available_elem_types(self):
        return onnx.TensorProto.DataType.values()

    def _verify_elem_type(self, elem_type):
        available_elem_types = self._get_available_elem_types()
        if elem_type not in available_elem_types:
            raise ValueError(
                f"Invalid elem_type: {elem_type}. Available values: {available_elem_types}. Check"
                "https://github.com/onnx/onnx/blob/96a0ca4374d2198944ff882bd273e64222b59cb9/onnx/onnx.proto3#L503-L551"
                "for details."
            )

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        self._verify_elem_type(config.source_dtype)
        self._verify_elem_type(config.target_dtype)

        source_dtype = ir.DataType(config.source_dtype)
        target_dtype = ir.DataType(config.target_dtype)

        ir_model = model.load_ir_model()

        pat = re.compile(config.name_pattern) if config.name_pattern else None

        wrapped_inputs = self._wrap_inputs(ir_model.graph, pat, source_dtype, target_dtype)
        wrapped_outputs = self._wrap_outputs(ir_model.graph, pat, source_dtype, target_dtype)
        if wrapped_inputs + wrapped_outputs == 0:
            logger.info("No inputs/outputs found with source_dtype=%s. Skip conversion.", source_dtype)
            return model
        logger.info(
            "Converted %d inputs and %d outputs from dtype=%s to dtype=%s",
            wrapped_inputs,
            wrapped_outputs,
            source_dtype,
            target_dtype,
        )

        ir_model.graph.sort()
        return ir_model_to_olive_model(ir_model, output_model_path, config)
