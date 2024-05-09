# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict, List

from collections import defaultdict
import onnx
from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam
import re

class OnnxIOFloat16ToFloat32(Pass):
    """Converts float16 model inputs/outputs to float32.

    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "name_pattern": PassConfigParam(
                type_=List[str], default_value="logits", description="Only convert inputs/outputs whose name matches this pattern"
            )
        }
        config.update(get_external_data_config())
        return config

    def create_io_mapping(graph, i_map, o_map):
        for n in graph.node:
            for i in n.input:
                i_map[i].append(n)
        for n in graph.node:
            for o in n.output:
                assert o not in o_map[o]
                o_map[o] = [n]

    def wrap_inputs(graph, i_map, names):
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
            print(f"input {i.name} from fp32")
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


    def wrap_outputs(graph, i_map, o_map, names):
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
            print(f"output {o.name} to fp32")
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

    def _run_for_config(
        self, model: ONNXModelHandler, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        ort_onnx_model = OnnxModel(model.load_model())

        i_map = defaultdict(list)
        o_map = defaultdict(list)

        self.create_io_mapping(model.graph, i_map, o_map)

        pat = None
        if args.name:
            pat = re.compile(args.name)

        self.wrap_inputs(model.graph, i_map, pat)
        self.wrap_outputs(model.graph, i_map, o_map, pat)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(ort_onnx_model.model, output_model_path, config)
