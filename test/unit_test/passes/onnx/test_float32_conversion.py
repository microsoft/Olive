# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
from onnx import TensorProto, helper

from olive.model.handler.onnx import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.float32_conversion import OnnxIOFloat16ToFloat32


def test_onnx_io_ft16_to_ft32_conversion(tmp_path):
    # setup
    node1 = helper.make_node("Add", ["logits_a", "logits_b"], ["logits_c"], name="add_node")

    input_tensor_a = helper.make_tensor_value_info("logits_a", TensorProto.FLOAT16, [None])
    input_tensor_b = helper.make_tensor_value_info("logits_b", TensorProto.FLOAT16, [None])
    output_tensor_c = helper.make_tensor_value_info("logits_c", TensorProto.FLOAT16, [None])

    graph = helper.make_graph([node1], "example_graph", [input_tensor_a, input_tensor_b], [output_tensor_c])
    onnx_model = helper.make_model(graph, producer_name="example_producer")
    tmp_model_path = str(tmp_path / "model.onnx")
    onnx.save(onnx_model, tmp_model_path)
    input_model = ONNXModelHandler(model_path=tmp_model_path)
    p = create_pass_from_dict(OnnxIOFloat16ToFloat32, None, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    output_model = p.run(input_model, None, output_folder)

    # assert
    for i in output_model.get_graph().input:
        assert i.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
    for o in output_model.get_graph().output:
        assert o.type.tensor_type.elem_type == onnx.TensorProto.FLOAT
