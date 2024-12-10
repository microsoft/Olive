# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper

from olive.model.handler.onnx import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries


def get_onnx_model(model_path):
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1])
    input2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1])

    helper.make_tensor_value_info("intermediate", TensorProto.FLOAT, [None])

    output = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [None])

    node1 = helper.make_node("Add", ["input1", "input2"], ["intermediate"], name="Add")
    node2 = helper.make_node("Relu", ["intermediate"], ["output1"], name="Relu")

    graph_def = helper.make_graph(
        [node1, node2],
        "test-model",
        [input1, input2],
        [output],
    )

    model = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path=str(model_path))


def get_quantized_model(model_path):
    input1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1])
    output = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1])

    scale_initializer = numpy_helper.from_array(np.array([0.1], dtype=np.float32), name="scale")
    zero_point_initializer = numpy_helper.from_array(np.array([128], dtype=np.uint8), name="zero_point")

    helper.make_tensor_value_info("quantized_output", TensorProto.UINT8, [1])
    quantize_node = helper.make_node(
        "QuantizeLinear", inputs=["input1", "scale", "zero_point"], outputs=["quantized_output"], name="QuantizeNode"
    )
    dequantize_node = helper.make_node(
        "DequantizeLinear",
        inputs=["quantized_output", "scale", "zero_point"],
        outputs=["output1"],
        name="DequantizeNode",
    )

    graph_def = helper.make_graph(
        [quantize_node, dequantize_node],
        "quantized-test-model",
        [input1],
        [output],
        initializer=[scale_initializer, zero_point_initializer],
    )

    model = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model, model_path)
    return ONNXModelHandler(model_path=str(model_path))


def test_rename_inputs(tmp_path):
    # setup
    renamed_input = "renamed_input"
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameInputs", "old_names": ["input1"], "new_names": [renamed_input]}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert renamed_input in [graph_input.name for graph_input in model_def.graph.input]


def test_rename_outputs(tmp_path):
    # setup
    renamed_output = "renamed_output"
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "RenameOutputs", "old_names": ["output1"], "new_names": [renamed_output]}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert renamed_output in [output.name for output in model_def.graph.output]


def test_remove_shapes(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_model = get_onnx_model(model_path)
    output_folder = str(tmp_path / "onnx")
    input_model = onnx.shape_inference.infer_shapes(input_model.load_model())
    assert len(input_model.graph.value_info) > 0
    onnx.save(input_model, model_path)
    p = create_pass_from_dict(GraphSurgeries, {"surgeries": [{"surgeon": "RemoveShapes"}]}, disable_search=True)

    # execute
    onnx_model = p.run(ONNXModelHandler(model_path=str(model_path)), output_folder)

    # assert
    assert len(onnx_model.load_model().graph.value_info) == 0


def test_infer_shapes(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(GraphSurgeries, {"surgeries": [{"surgeon": "InferShapes"}]}, disable_search=True)

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    for output in onnx_model.load_model().graph.output:
        if output.name == "output1":
            tensor_type = output.type.tensor_type
            shape = tensor_type.shape
            dims = [dim.dim_value for dim in shape.dim]
            assert dims == [1]


def test_remove_initializer_from_inputs(tmp_path):
    # setup
    model_path = tmp_path / "model.onnx"
    input_tensor = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3])
    initializer_tensor = helper.make_tensor(
        name="const1",
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=[1.0, 2.0, 3.0],
    )
    node = helper.make_node(
        "Add",
        inputs=["input1", "const1"],
        outputs=["output1"],
    )
    output_tensor = helper.make_tensor_value_info("output1", TensorProto.FLOAT, [1, 3])
    graph = helper.make_graph(
        nodes=[node],
        name="TestGraph",
        inputs=[input_tensor, helper.make_tensor_value_info("const1", TensorProto.FLOAT, [1, 3])],
        outputs=[output_tensor],
        initializer=[initializer_tensor],
    )
    model = helper.make_model(graph)
    onnx.save(model, model_path)

    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "RemoveInitializerFromInputs"}]}, disable_search=True
    )

    # execute
    output_model = p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "onnx"))

    # assert
    assert Path(output_model.model_path).exists()
    output_model = output_model.load_model()
    assert "const1" not in {graph_input.name for graph_input in output_model.graph.input}


def test_reorder_inputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "ReorderInputs", "permutation": [1, 0]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [graph_input.name for graph_input in model_def.graph.input] == ["input2", "input1"]


def test_zero_out_input(tmp_path):
    # setup
    input_model_path = tmp_path / "model.onnx"
    input_model = get_onnx_model(input_model_path)
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ZeroOutInput", "node_name": "Add", "input_idx": 1}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    add_node = next(node for node in model_def.graph.node if node.name == "Add")
    zero_node = next(node for node in model_def.graph.node if node.name == "Add_zero")
    assert add_node.input[1] == zero_node.output[0]

    zero_tensor = zero_node.attribute[0].t
    zero_values = np.array(zero_tensor.float_data, dtype=np.float32).reshape(zero_tensor.dims)
    assert np.all(zero_values == 0), "Zero tensor should contain all zeros."


def test_remove_inputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "RemoveInputs", "names": ["input1"]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [graph_input.name for graph_input in model_def.graph.input] == ["input2"]
    for node in model_def.graph.node:
        assert "input1" not in node.input


def test_expose_outputs(tmp_path):
    # setup
    input_model = get_onnx_model(tmp_path / "model.onnx")
    output_folder = str(tmp_path / "onnx")
    p = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "ExposeOutputs", "names": ["Add"]}]}, disable_search=True
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    model_def = onnx_model.load_model()
    assert [output.name for output in model_def.graph.output] == ["output1", "intermediate"]


def test_expose_quantized_output(tmp_path):
    # setup
    input_model_path = tmp_path / "quantized_model.onnx"
    input_model = get_quantized_model(input_model_path)
    output_folder = str(tmp_path / "onnx")

    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": "ExposeQuantizedOutput", "output_name": "output1"}]},
        disable_search=True,
    )

    # execute
    onnx_model = p.run(input_model, output_folder)

    # assert
    output_model = onnx_model.load_model()

    # Retrieve the quantized output name from the original model
    dequantized_node = next(node for node in input_model.load_model().graph.node if node.op_type == "DequantizeLinear")
    quantized_output_name = dequantized_node.input[0]

    # Retrieve the scale and zero_point initializer names from the original model
    quantized_node = next(node for node in input_model.load_model().graph.node if quantized_output_name in node.output)
    scale_name_in_original_model = quantized_node.input[1]
    zero_point_name_in_original_model = quantized_node.input[2]

    # Retrieve the scale and zero_point values from the original model
    original_scale_initializer = next(
        init for init in input_model.load_model().graph.initializer if init.name == scale_name_in_original_model
    )
    original_scale_value = numpy_helper.to_array(original_scale_initializer)[0]

    original_zero_point_initializer = next(
        init for init in input_model.load_model().graph.initializer if init.name == zero_point_name_in_original_model
    )
    original_zero_point_value = numpy_helper.to_array(original_zero_point_initializer)[0]
    zero_point_dtype = helper.tensor_dtype_to_np_dtype(original_zero_point_initializer.data_type)

    # Validate that the quantized output is exposed in the modified model
    exposed_outputs = [output.name for output in output_model.graph.output]
    assert quantized_output_name in exposed_outputs, "Quantized output not exposed."
    assert "output1" not in exposed_outputs, "Original output should be removed."

    # Construct expected node and initializer names for scale and zero_point
    output_name = "output1"
    scale_node_name = f"{output_name}_exposed_scale"
    scale_initializer_name = f"{scale_node_name}_value"

    zero_point_node_name = f"{output_name}_exposed_zero_point"
    zero_point_initializer_name = f"{zero_point_node_name}_value"

    # Validate that the scale node and its initializer exist in the modified model
    assert any(node.name == scale_node_name for node in output_model.graph.node), "Scale node not added."
    scale_initializer = next(init for init in output_model.graph.initializer if init.name == scale_initializer_name)
    assert np.allclose(
        numpy_helper.to_array(scale_initializer), np.array([original_scale_value], dtype=np.float32)
    ), "Scale value mismatch."

    # Validate that the zero_point node and its initializer exist in the modified model
    assert any(node.name == zero_point_node_name for node in output_model.graph.node), "Zero point node not added."
    zero_point_initializer = next(
        init for init in output_model.graph.initializer if init.name == zero_point_initializer_name
    )
    assert np.allclose(
        numpy_helper.to_array(zero_point_initializer), np.array([original_zero_point_value], dtype=zero_point_dtype)
    ), "Zero point value mismatch."
