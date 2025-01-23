# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Optional, Type

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxIODataTypeConverter(Pass):
    """Converts model inputs/outputs from a source dtype to a target dtype based on a name pattern."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
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

    def create_io_mapping(self, graph, i_map, o_map):
        for n in graph.node:
            for i in n.input:
                i_map[i].append(n)
        for n in graph.node:
            for o in n.output:
                assert o not in o_map[o]
                o_map[o] = [n]

    def wrap_inputs(self, graph, i_map, names, source_dtype, target_dtype) -> int:
        # 1. find source_dtype inputs
        # 2. rewrite all consumers
        # 3. insert cast
        # 4. rewrite graph inputs
        inputs = [n for n in graph.input if n.type.tensor_type.elem_type == source_dtype]
        converted_count = 0
        for i in inputs:
            if not self._is_name_matched(i.name, names):
                continue
            logger.debug("Converting input %s from %s to %s", i.name, source_dtype, target_dtype)
            for n in i_map[i.name]:
                for j, o in enumerate(n.input):
                    if o == i.name:
                        n.input[j] = i.name + "_converted"

            cast = onnx.helper.make_node("Cast", inputs=[i.name], outputs=[i.name + "_converted"], to=source_dtype)

            graph.node.insert(0, cast)
            i.type.tensor_type.elem_type = target_dtype
            converted_count += 1

        return converted_count

    def wrap_outputs(self, graph, i_map, o_map, names, source_dtype, target_dtype) -> int:
        # 1. find source dtype outputs
        # 2. rewrite all providers
        # 3. append cast
        # 4. rewrite graph outputs
        outputs = [n for n in graph.output if n.type.tensor_type.elem_type == source_dtype]
        converted_count = 0
        for o in outputs:
            if not self._is_name_matched(o.name, names):
                continue
            logger.debug("Converting output %s from %s to %s", o.name, source_dtype, target_dtype)
            for n in o_map[o.name]:
                for j, i_ in enumerate(n.output):
                    if i_ == o.name:
                        n.output[j] = o.name + "_converted"
            for n in i_map[o.name]:
                for j, i_ in enumerate(n.input):
                    if i_ == o.name:
                        n.input[j] = o.name + "_converted"

            cast = onnx.helper.make_node(
                "Cast",
                inputs=[o.name + "_converted"],
                outputs=[o.name],
                to=target_dtype,
            )
            graph.node.append(cast)
            o.type.tensor_type.elem_type = target_dtype
            converted_count += 1

        return converted_count

    def _is_name_matched(self, name: str, names: Optional[re.Pattern]) -> bool:
        return not names or bool(names.search(name))

    @staticmethod
    def get_elem_type_from_number(num):
        for value in vars(onnx.TensorProto).values():
            if isinstance(value, int) and value == num:
                return value
        raise ValueError(f"Invalid elem_type number: {num}")

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
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        ort_onnx_model = OnnxModel(model.load_model())

        i_map = defaultdict(list)
        o_map = defaultdict(list)

        self.create_io_mapping(ort_onnx_model.model.graph, i_map, o_map)

        pat = None
        if config.name_pattern:
            pat = re.compile(config.name_pattern)

        source_dtype = config.source_dtype
        target_dtype = config.target_dtype

        self._verify_elem_type(source_dtype)
        self._verify_elem_type(target_dtype)

        source_dtype = self.get_elem_type_from_number(source_dtype)
        target_dtype = self.get_elem_type_from_number(target_dtype)

        wrapped_inputs = self.wrap_inputs(ort_onnx_model.model.graph, i_map, pat, source_dtype, target_dtype)
        wrapped_outputs = self.wrap_outputs(ort_onnx_model.model.graph, i_map, o_map, pat, source_dtype, target_dtype)
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

        return model_proto_to_olive_model(ort_onnx_model.model, output_model_path, config)
