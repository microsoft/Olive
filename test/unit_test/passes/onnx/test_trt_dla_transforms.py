# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.tensorrt.trt_dla_transforms import TrtMatMulToConvTransform


def create_model_with_matmul(tmp_path, input_shape, weight_shape):
    """Create a test model with a MatMul operation and non-4D tensors."""
    model_path = tmp_path / "matmul_model.onnx"

    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    weight_data = np.random.randn(*weight_shape).astype(np.float32)
    weight_initializer = helper.make_tensor(
        name="weight",
        data_type=TensorProto.FLOAT,
        dims=weight_shape,
        vals=weight_data.flatten().tolist(),
    )

    output_shape = input_shape[:-1] + [weight_shape[-1]]
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    matmul_node = helper.make_node("MatMul", inputs=["input", "weight"], outputs=["output"], name="MatMul_Node")

    graph = helper.make_graph(
        nodes=[matmul_node],
        name="MatMul-Test-Graph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[weight_initializer],
    )

    model = helper.make_model(graph, producer_name="onnx-example")
    model.opset_import[0].version = 13

    onnx.save(model, model_path)

    return str(model_path), input_shape, weight_shape, output_shape


def test_matmul_to_conv_transform(tmp_path):
    input_shape = [1, 3136, 512]  # A shape
    weight_shape = [512, 128]  # B shape

    model_path, _, _, _ = create_model_with_matmul(tmp_path, input_shape, weight_shape)
    input_model = ONNXModelHandler(model_path=model_path)
    output_folder = "output"

    p = create_pass_from_dict(TrtMatMulToConvTransform, {}, disable_search=True)
    output_model = p.run(input_model, output_folder)

    transformed_model = output_model.load_model()

    # Verify MatMul is replaced
    matmul_nodes = [node for node in transformed_model.graph.node if node.op_type == "MatMul"]
    assert len(matmul_nodes) == 0, "MatMul operation should be replaced"

    # Verify Transpose-Conv-Transpose sequence exists
    transpose_nodes = [node for node in transformed_model.graph.node if node.op_type == "Transpose"]
    conv_nodes = [node for node in transformed_model.graph.node if node.op_type == "Conv"]

    assert len(transpose_nodes) >= 2, "Expected at least 2 Transpose operations"
    assert len(conv_nodes) == 1, "Expected 1 Conv operation"

    # Verify numerical correctness with random input
    input_data = np.random.randn(*input_shape).astype(np.float32)

    # Create session options with optimization disabled
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL

    # Run inference on both original and transformed models
    original_session = ort.InferenceSession(model_path, session_options)
    original_result = original_session.run(None, {"input": input_data})[0]

    transformed_session = ort.InferenceSession(output_model.model_path, session_options)
    transformed_result = transformed_session.run(None, {"input": input_data})[0]

    # Verify output shape is as expected
    assert original_result.shape == transformed_result.shape, "Output shapes should match"

    # Verify numerical equivalence
    np.testing.assert_allclose(original_result, transformed_result, rtol=1e-4, atol=1e-4)
