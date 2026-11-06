# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries


@pytest.mark.parametrize("input_shape", [(2, 3), (2, 2, 3)])
@pytest.mark.parametrize("output_names", [("y",), ("product", "y"), ("y", "product"), ("biased", "y")])
@pytest.mark.parametrize("with_relu", [False, True])
def test_matmul_add_to_gemm_preserves_public_outputs(input_shape, output_names, with_relu, tmp_path):
    input_data = np.arange(np.prod(input_shape), dtype=np.float32).reshape(input_shape) - 3
    weight = np.arange(12, dtype=np.float32).reshape(3, 4) - 5
    bias = np.array([1, -2, 3, -4], dtype=np.float32)
    product = input_data @ weight
    expected = {
        "product": product,
        "biased": product + bias,
        "y": np.maximum(product + bias, 0) if with_relu else product + bias,
    }
    nodes = [
        helper.make_node("MatMul", ["x", "weight"], ["product"], name="MatMul"),
        helper.make_node("Add", ["product", "bias"], ["biased"], name="Add"),
    ]
    final_input = "biased"
    if with_relu:
        nodes.append(helper.make_node("Relu", [final_input], ["activated"], name="Relu"))
        final_input = "activated"
    nodes.append(helper.make_node("Identity", [final_input], ["y"], name="Identity"))
    graph = helper.make_graph(
        nodes,
        "public_outputs",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)],
        [helper.make_tensor_value_info(name, TensorProto.FLOAT, expected[name].shape) for name in output_names],
        [numpy_helper.from_array(weight, name="weight"), numpy_helper.from_array(bias, name="bias")],
    )
    original = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)], ir_version=8)
    onnx.checker.check_model(original, full_check=True)
    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    original_session = ort.InferenceSession(
        original.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for name, actual in zip(output_names, original_session.run(list(output_names), {"x": input_data})):
        np.testing.assert_array_equal(actual, expected[name])

    model_path = tmp_path / "model.onnx"
    onnx.save(original, model_path)
    surgery = create_pass_from_dict(
        GraphSurgeries, {"surgeries": [{"surgeon": "MatMulAddToGemm"}]}, disable_search=True
    )
    output_model = surgery.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "output"))
    rewritten = output_model.load_model()
    onnx.checker.check_model(rewritten, full_check=True)
    assert [value.SerializeToString() for value in rewritten.graph.output] == [
        value.SerializeToString() for value in original.graph.output
    ]
    rewritten_session = ort.InferenceSession(
        rewritten.SerializeToString(), sess_options=options, providers=["CPUExecutionProvider"]
    )
    for name, actual in zip(output_names, rewritten_session.run(list(output_names), {"x": input_data})):
        np.testing.assert_array_equal(actual, expected[name])
    op_types = [node.op_type for node in rewritten.graph.node]
    assert ("Gemm" in op_types) == (output_names == ("y",))
    assert ("MatMul" in op_types) == (output_names != ("y",))
