# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model
from typing import Any, Dict

from onnx import TensorProto, helper

from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR
from olive.model import ONNXModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx import OnnxModelOptimizer
from olive.passes.onnx.common import model_proto_to_olive_model


def test_onnx_model_optimizer_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxModelOptimizer, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, None, output_folder)


def _make_model_for_patch_unsupported_argmax_operator(
    data_type: TensorProto.DataType, filepath: str, config: Dict[str, Any]
) -> ONNXModel:
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
        producer_name="From test_model_optimizer.py",
        opset_imports=opset_imports,
    )

    return model_proto_to_olive_model(model, filepath, config)


def test_onnx_model_optimizer_pass_patch_unsupported_argmax_operator_modified(tmp_path):
    pass_config = {"save_as_external_data": False, "all_tensors_to_one_file": False, "external_data_name": False}

    m = _make_model_for_patch_unsupported_argmax_operator(TensorProto.INT64, str(tmp_path / "input.onnx"), pass_config)
    p = create_pass_from_dict(
        OnnxModelOptimizer, pass_config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR
    )

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


def test_onnx_model_optimizer_pass_patch_unsupported_argmax_operator_unmodified(tmp_path):
    pass_config = {"save_as_external_data": False, "all_tensors_to_one_file": False, "external_data_name": False}

    m = _make_model_for_patch_unsupported_argmax_operator(TensorProto.INT32, str(tmp_path / "input.onnx"), pass_config)
    p = create_pass_from_dict(
        OnnxModelOptimizer, pass_config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR
    )

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


def test_onnx_model_optimizer_pass_fuse_reshape_operations(tmp_path):
    import numpy as np

    X = helper.make_tensor_value_info("X", TensorProto.INT64, [None])  # noqa: N806
    Y = helper.make_tensor_value_info("Y", TensorProto.INT64, [None])  # noqa: N806

    node_def_1 = helper.make_node("Reshape", ["X", "shape_1"], ["Z"], domain="com.microsoft")
    node_def_2 = helper.make_node("Reshape", ["Z", "shape_2"], ["Y"], domain="com.microsoft")

    shape_data_1 = np.array([2, 4, 6], np.int64)
    shape_init_1 = helper.make_tensor(
        name="shape_1",
        data_type=TensorProto.INT64,
        dims=shape_data_1.shape,
        vals=shape_data_1.tobytes(),
        raw=True,
    )

    shape_data_2 = np.array([-1], np.int64)
    shape_init_2 = helper.make_tensor(
        name="shape_2",
        data_type=TensorProto.INT64,
        dims=shape_data_2.shape,
        vals=shape_data_2.tobytes(),
        raw=True,
    )

    graph_def = helper.make_graph(
        [node_def_1, node_def_2],
        "g",
        [X],
        [Y],
        initializer=[shape_init_1, shape_init_2],
    )
    opset_imports = [
        helper.make_operatorsetid("", 18),
        helper.make_operatorsetid("com.microsoft", 1),
    ]
    model = helper.make_model(
        graph_def,
        producer_name="From test_model_optimizer.py",
        opset_imports=opset_imports,
    )

    pass_config = {"save_as_external_data": False, "all_tensors_to_one_file": False, "external_data_name": False}

    m = model_proto_to_olive_model(model, str(tmp_path / "input.onnx"), pass_config)
    p = create_pass_from_dict(
        OnnxModelOptimizer, pass_config, disable_search=True, accelerator_spec=DEFAULT_CPU_ACCELERATOR
    )

    actual_model = p.run(m, None, str(tmp_path / "onnx"))
    assert Path(actual_model.model_path).exists()

    actual_model = actual_model.load_model()
    assert len(actual_model.graph.node) == 1

    reshape_op_count = 0
    others_op_count = 0
    for node in actual_model.graph.node:
        if node.op_type == "Reshape":
            reshape_op_count += 1
        else:
            others_op_count += 1

    assert reshape_op_count == 1
    assert others_op_count == 0
