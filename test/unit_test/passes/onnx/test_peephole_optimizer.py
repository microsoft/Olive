# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from test.unit_test.utils import get_onnx_model
from typing import TYPE_CHECKING, Any, Dict

import pytest
from onnx import TensorProto, helper

from olive.hardware import DEFAULT_CPU_ACCELERATOR, DEFAULT_GPU_CUDA_ACCELERATOR
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.common import model_proto_to_olive_model
from olive.passes.onnx.peephole_optimizer import OnnxPeepholeOptimizer

if TYPE_CHECKING:
    from olive.model import ONNXModelHandler


@pytest.fixture(name="external_data_config")
def external_data_config_fixture():
    return {
        "save_as_external_data": False,
        "all_tensors_to_one_file": True,
        "external_data_name": None,
        "size_threshold": 1024,
        "convert_attribute": False,
    }


def test_onnx_peephole_optimizer_pass(tmp_path):
    # setup
    input_model = get_onnx_model()
    p = create_pass_from_dict(OnnxPeepholeOptimizer, {}, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    p.run(input_model, output_folder)


def _make_model_for_patch_unsupported_argmax_operator(
    data_type: TensorProto.DataType, filepath: str, config: Dict[str, Any]
) -> "ONNXModelHandler":
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
        producer_name="From test_peephole_optimizer.py",
        opset_imports=opset_imports,
    )

    return model_proto_to_olive_model(model, filepath, config)


def test_onnx_peephole_optimizer_pass_patch_unsupported_argmax_operator_modified(tmp_path, external_data_config):
    m = _make_model_for_patch_unsupported_argmax_operator(
        TensorProto.INT64, str(tmp_path / "input.onnx"), external_data_config
    )
    p = create_pass_from_dict(
        OnnxPeepholeOptimizer, external_data_config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR
    )

    actual_model = p.run(m, str(tmp_path / "onnx"))
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


def test_onnx_peephole_optimizer_pass_patch_unsupported_argmax_operator_unmodified(tmp_path, external_data_config):
    m = _make_model_for_patch_unsupported_argmax_operator(
        TensorProto.INT32, str(tmp_path / "input.onnx"), external_data_config
    )
    p = create_pass_from_dict(
        OnnxPeepholeOptimizer, external_data_config, disable_search=True, accelerator_spec=DEFAULT_GPU_CUDA_ACCELERATOR
    )

    actual_model = p.run(m, str(tmp_path / "onnx"))
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


# TODO(team): this test will creat an unnecessary intermediate model file. Need to optimize it.
def test_onnx_peephole_optimizer_pass_fuse_reshape_operations(tmp_path, external_data_config):
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
        producer_name="From test_peephole_optimizer.py",
        opset_imports=opset_imports,
    )

    m = model_proto_to_olive_model(model, str(tmp_path / "input.onnx"), external_data_config)
    p = create_pass_from_dict(
        OnnxPeepholeOptimizer, external_data_config, disable_search=True, accelerator_spec=DEFAULT_CPU_ACCELERATOR
    )

    actual_model = p.run(m, str(tmp_path / "onnx"))
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


def test_onnx_peephole_optimizer_pass_constant_folding(tmp_path, external_data_config):
    import numpy as np

    # setup
    model = _get_onnx_model_with_constant(str(tmp_path / "input.onnx"), external_data_config)
    input_model = model.load_model()
    input_constant_nodes = [node for node in input_model.graph.node if node.op_type == "Constant"]
    input_initializer_names = {initializer.name for initializer in input_model.graph.initializer}
    p = create_pass_from_dict(
        OnnxPeepholeOptimizer, None, disable_search=True, accelerator_spec=DEFAULT_CPU_ACCELERATOR
    )

    # execute
    output_model = p.run(model, tmp_path / "onnx")

    # assert
    assert Path(output_model.model_path).exists()
    output_model = output_model.load_model()
    output_constant_nodes = [node for node in output_model.graph.node if node.op_type == "Constant"]
    output_initializer_names = {initializer.name for initializer in output_model.graph.initializer}
    assert len(output_constant_nodes) < len(input_constant_nodes), "Constant nodes were not folded."
    assert len(output_initializer_names) > len(input_initializer_names), "Initializers were not updated."
    inputs = {"input": np.random.rand(1, 3).astype(np.float32)}
    original_outputs = _run_onnx_model(input_model, inputs)
    optimized_outputs = _run_onnx_model(output_model, inputs)
    assert np.allclose(
        original_outputs, optimized_outputs, atol=1e-6
    ), "Outputs are not consistent after constant folding."


def _get_onnx_model_with_constant(model_path, external_data_config):
    input_tensor = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3])
    output_tensor = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3])
    const_tensor = helper.make_tensor(
        name="const_add",
        data_type=TensorProto.FLOAT,
        dims=[1, 3],
        vals=[1.0, 2.0, 3.0],
    )
    const_node = helper.make_node(
        "Constant",
        inputs=[],
        outputs=["const"],
        value=const_tensor,
    )
    add_node = helper.make_node(
        "Add",
        inputs=["input", "const"],
        outputs=["output"],
    )
    graph_def = helper.make_graph(
        nodes=[const_node, add_node],
        name="ConstantFoldingGraph",
        inputs=[input_tensor],
        outputs=[output_tensor],
        initializer=[],
    )
    opset_imports = [helper.make_operatorsetid("", 21)]
    model = helper.make_model(graph_def, producer_name="onnx-example", opset_imports=opset_imports)
    return model_proto_to_olive_model(model, model_path, external_data_config)


def _run_onnx_model(model, inputs):
    import onnxruntime as ort

    session = ort.InferenceSession(model.SerializeToString(), providers=["CPUExecutionProvider"])
    return session.run(None, inputs)
