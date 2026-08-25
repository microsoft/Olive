# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import onnx
import onnxruntime as ort
import pytest
from onnx import TensorProto, helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.float16_conversion import OnnxFloatToFloat16
from test.utils import get_onnx_model


@pytest.mark.parametrize("keep_io_types", [True, False])
def test_onnxfloattofloat16(keep_io_types, tmp_path):
    # setup
    # this is a simple model with a single Gemm node
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxFloatToFloat16, {"keep_io_types": keep_io_types}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    output_model = p.run(input_model, output_folder)

    # assert
    # check that the input and output types are as expected
    io_config = output_model.io_config
    for io_type in [*io_config["input_types"], *io_config["output_types"]]:
        assert io_type == ("float32" if keep_io_types else "float16")

    # check that the model initializer types are float16
    for initializer in output_model.load_model().graph.initializer:
        assert initializer.data_type == onnx.TensorProto.FLOAT16


def test_onnxfloattofloat16_preserves_cascaded_casts_in_subgraph(tmp_path):
    body = helper.make_graph(
        [
            helper.make_node("Cast", ["body_input"], ["as_int"], to=TensorProto.INT64),
            helper.make_node("Cast", ["as_int"], ["body_output"], to=TensorProto.FLOAT),
        ],
        "scan_body",
        [helper.make_tensor_value_info("body_input", TensorProto.FLOAT, [])],
        [helper.make_tensor_value_info("body_output", TensorProto.FLOAT, [])],
    )
    model = helper.make_model(
        helper.make_graph(
            [helper.make_node("Scan", ["input"], ["output"], body=body, num_scan_inputs=1)],
            "scan_model",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, [None])],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None])],
        ),
        opset_imports=[helper.make_opsetid("", 18)],
    )
    input_path = tmp_path / "scan.onnx"
    onnx.save(model, input_path)

    olive_pass = create_pass_from_dict(OnnxFloatToFloat16, {}, disable_search=True)
    output_model = olive_pass.run(ONNXModelHandler(model_path=input_path), str(tmp_path / "output"))
    session = ort.InferenceSession(str(output_model.model_path), providers=["CPUExecutionProvider"])

    result = session.run(None, {"input": np.array([1.5, -1.5], dtype=np.float16)})[0]
    np.testing.assert_array_equal(result, np.array([1.0, -1.0], dtype=np.float16))
