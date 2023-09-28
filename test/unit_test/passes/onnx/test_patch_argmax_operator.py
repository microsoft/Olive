# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Any, Dict

from onnx import TensorProto, helper

from olive.model import ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.onnx.patch_argmax_operator import OrtPatchArgMaxOperator


def _make_model(data_type: TensorProto.DataType, filepath: str, config: Dict[str, Any]) -> ONNXModel:
    X = helper.make_tensor_value_info("X", data_type, [None, None])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", data_type, [None])  # noqa: N806
    node_def = helper.make_node("ArgMax", ["X"], ["Y"], domain="com.microsoft")
    graph_def = helper.make_graph(
        [node_def],
        "g",
        [X],
        [Y],
    )
    opset_imports = [
        helper.make_operatorsetid("", 18),
        helper.make_operatorsetid("com.microsoft", 1),
    ]
    model = helper.make_model(
        graph_def,
        producer_name="From test_patch_argmax_operator.py",
        opset_imports=opset_imports,
    )

    return model_proto_to_olive_model(model, filepath, config)


def test_patch_argmax_operator_pass_does_insert(tmp_path):
    config = {"save_as_external_data": False, "all_tensors_to_one_file": False, "external_data_name": False}

    m = _make_model(TensorProto.INT64, str(tmp_path / "input.onnx"), config)
    p = create_pass_from_dict(OrtPatchArgMaxOperator, config, disable_search=True)

    # execute
    actual_model = p.run(m, None, str(tmp_path / "onnx"))
    assert Path(actual_model.model_path).exists()

    actual_model = actual_model.load_model()
    assert len(actual_model.graph.node) == 2

    argmax_op_count = 0
    cast_op_count = 0
    others_op_count = 0
    for node in actual_model.graph.node:
        if node.op_type == "ArgMax":
            argmax_op_count += 1
        elif node.op_type == "Cast":
            cast_op_count += 1
        else:
            others_op_count += 1

    assert argmax_op_count == 1
    assert cast_op_count == 1
    assert others_op_count == 0


def test_patch_argmax_operator_pass_no_insert(tmp_path):
    config = {"save_as_external_data": False, "all_tensors_to_one_file": False, "external_data_name": False}

    m = _make_model(TensorProto.INT32, str(tmp_path / "input.onnx"), config)
    p = create_pass_from_dict(OrtPatchArgMaxOperator, config, disable_search=True)

    # execute
    actual_model = p.run(m, None, str(tmp_path / "onnx"))
    assert Path(actual_model.model_path).exists()

    actual_model = actual_model.load_model()
    assert len(actual_model.graph.node) == 1

    argmax_op_count = 0
    others_op_count = 0
    for node in actual_model.graph.node:
        if node.op_type == "ArgMax":
            argmax_op_count += 1
        else:
            others_op_count += 1

    assert argmax_op_count == 1
    assert others_op_count == 0
