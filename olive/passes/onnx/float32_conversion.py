# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict

import onnx

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

logger = logging.getLogger(__name__)


class OnnxIOFloat16ToFloat32(Pass):
    """Converts float16 model inputs/outputs to float32."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "name_pattern": PassConfigParam(
                type_=str,
                default_value="logits",
                description=(
                    "Only convert inputs/outputs whose name matches this pattern. By defaultlooking for logits names"
                ),
            )
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

    def wrap_inputs(self, graph, i_map, names) -> int:
        # 1. find fp16 inputs
        # 2. rewrite all consumers
        # 3. insert cast
        # 4. rewrite graph inputs
        inputs = [n for n in graph.input if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
        for i in inputs:
            if names:
                match = names.search(i.name)
                if not match:
                    continue
            logger.debug("input %s from fp32", i.name)
            for n in i_map[i.name]:
                for j, o in enumerate(n.input):
                    if o == i.name:
                        n.input[j] = i.name + "_fp16"
            cast = onnx.helper.make_node(
                "Cast",
                inputs=[i.name],
                outputs=[i.name + "_fp16"],
                to=onnx.TensorProto.FLOAT16,
            )
            graph.node.insert(0, cast)
            i.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        return len(inputs)

    def wrap_outputs(self, graph, i_map, o_map, names) -> int:
        # 1. find fp16 outputs
        # 2. rewrite all providers
        # 3. append cast
        # 4. rewrite graph outputs
        outputs = [n for n in graph.output if n.type.tensor_type.elem_type == onnx.TensorProto.FLOAT16]
        for o in outputs:
            if names:
                match = names.search(o.name)
                if not match:
                    continue
            logger.debug("output %s from fp32", o.name)
            for n in o_map[o.name]:
                for j, i in enumerate(n.output):
                    if i == o.name:
                        n.output[j] = o.name + "_fp16"
            for n in i_map[o.name]:
                for j, i in enumerate(n.input):
                    if i == o.name:
                        n.input[j] = o.name + "_fp16"

            cast = onnx.helper.make_node(
                "Cast",
                inputs=[o.name + "_fp16"],
                outputs=[o.name],
                to=onnx.TensorProto.FLOAT,
            )
            graph.node.append(cast)
            o.type.tensor_type.elem_type = onnx.TensorProto.FLOAT

        return len(outputs)

    def _run_for_config(
        self, model: ONNXModelHandler, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        ort_onnx_model = OnnxModel(model.load_model())

        i_map = defaultdict(list)
        o_map = defaultdict(list)

        self.create_io_mapping(ort_onnx_model.model.graph, i_map, o_map)

        pat = None
        if config["name_pattern"]:
            pat = re.compile(config["name_pattern"])

        wrapped_inputs = self.wrap_inputs(ort_onnx_model.model.graph, i_map, pat)
        wrapped_outputs = self.wrap_outputs(ort_onnx_model.model.graph, i_map, o_map, pat)
        if wrapped_inputs + wrapped_outputs == 0:
            logger.info("No float16 inputs/outputs found. Skip conversion.")
            return model
        logger.info("Converted %d inputs and %d outputs from float16 to float32", wrapped_inputs, wrapped_outputs)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(ort_onnx_model.model, output_model_path, config)
