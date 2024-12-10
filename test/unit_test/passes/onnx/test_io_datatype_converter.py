# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
import pytest
from onnx import helper

from olive.model.handler.onnx import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.io_datatype_converter import OnnxIODataTypeConverter


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [
        (1, 10),
        (10, 1),
    ],
)
def test_onnx_io_datatype_conversion(source_dtype, target_dtype, tmp_path):
    # setup
    node1 = helper.make_node("Add", ["logits_a", "logits_b"], ["logits_c"], name="add_node")

    input_tensor_a = helper.make_tensor_value_info("logits_a", source_dtype, [None])
    input_tensor_b = helper.make_tensor_value_info("logits_b", source_dtype, [None])
    output_tensor_c = helper.make_tensor_value_info("logits_c", source_dtype, [None])

    graph = helper.make_graph([node1], "example_graph", [input_tensor_a, input_tensor_b], [output_tensor_c])
    onnx_model = helper.make_model(graph, producer_name="example_producer")
    tmp_model_path = str(tmp_path / "model.onnx")
    onnx.save(onnx_model, tmp_model_path)
    input_model = ONNXModelHandler(model_path=tmp_model_path)
    p = create_pass_from_dict(
        OnnxIODataTypeConverter, {"source_dtype": source_dtype, "target_dtype": target_dtype}, disable_search=True
    )
    output_folder = str(tmp_path / "onnx")

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    for i in output_model.get_graph().input:
        assert i.type.tensor_type.elem_type == target_dtype
    for o in output_model.get_graph().output:
        assert o.type.tensor_type.elem_type == target_dtype


@pytest.mark.parametrize(
    ("source_dtype", "target_dtype"),
    [
        (-1, 10),
        (10, 100),
    ],
)
def test_onnx_io_datatype_converter_invalid_datatype(source_dtype, target_dtype, tmp_path):
    # setup
    node1 = helper.make_node("Add", ["logits_a"], ["logits_b"], name="add_node")

    input_tensor_a = helper.make_tensor_value_info("logits_a", source_dtype, [None])
    output_tensor_b = helper.make_tensor_value_info("logits_b", source_dtype, [None])

    graph = helper.make_graph([node1], "example_graph", [input_tensor_a], [output_tensor_b])
    onnx_model = helper.make_model(graph, producer_name="example_producer")
    tmp_model_path = str(tmp_path / "model.onnx")
    onnx.save(onnx_model, tmp_model_path)
    input_model = ONNXModelHandler(model_path=tmp_model_path)
    p = create_pass_from_dict(
        OnnxIODataTypeConverter, {"source_dtype": source_dtype, "target_dtype": target_dtype}, disable_search=True
    )
    output_folder = str(tmp_path / "onnx")

    # execute
    with pytest.raises(ValueError, match="Invalid elem_type"):
        p.run(input_model, output_folder)
