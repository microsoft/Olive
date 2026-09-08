# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from onnxruntime import InferenceSession

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.graph_surgery.activations import FuseBiasGelu, FuseGelu
from olive.passes.onnx.graph_surgery.base import Surgeon

_SQRT_2 = math.sqrt(2.0)


def _run_surgery(model, tmp_path, surgeon, name):
    model_path = tmp_path / f"{name}.onnx"
    onnx.save(model, model_path)
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": surgeon}], "remove_duplicate_initializers": False},
        disable_search=True,
    )
    output_model = graph_surgeries.run(ONNXModelHandler(model_path=str(model_path)), tmp_path / f"{name}_output")
    output = output_model.load_model()
    onnx.checker.check_model(output)
    return output


def _count_ops(model):
    return {
        op_type: sum(node.op_type == op_type for node in model.graph.node)
        for op_type in {n.op_type for n in model.graph.node}
    }


def _make_model(nodes, initializers, outputs, *, input_shape=(1, 4, 8), opset=21):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, input_shape)
    graph_outputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, input_shape) for name in outputs]
    graph = helper.make_graph(nodes, "activation_test", [x], graph_outputs, initializers)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", opset)])


def _exact_gelu_parts(*, input_name="x", half_first=False, sqrt_2=_SQRT_2):
    initializers = [
        numpy_helper.from_array(np.array(sqrt_2, dtype=np.float32), name="sqrt_2"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half"),
    ]
    nodes = [
        helper.make_node("Div", [input_name, "sqrt_2"], ["divided"]),
        helper.make_node("Erf", ["divided"], ["erf"]),
        helper.make_node("Add", ["erf", "one"], ["shifted"]),
        helper.make_node("Mul", [input_name, "shifted"], ["scaled"]),
        helper.make_node("Mul", ["half", "scaled"] if half_first else ["scaled", "half"], ["y"]),
    ]
    return nodes, initializers


def _build_exact_gelu(*, half_first=False, sqrt_2=_SQRT_2, opset=21):
    nodes, initializers = _exact_gelu_parts(half_first=half_first, sqrt_2=sqrt_2)
    return _make_model(nodes, initializers, ["y"], opset=opset)


def _build_approximate_gelu(*, half_first=False, coefficient=0.044715, input_name="x"):
    initializers = [
        numpy_helper.from_array(np.array(3.0, dtype=np.float32), name="three"),
        numpy_helper.from_array(np.array(coefficient, dtype=np.float32), name="coefficient"),
        numpy_helper.from_array(np.array(math.sqrt(2.0 / math.pi), dtype=np.float32), name="sqrt_2_over_pi"),
        numpy_helper.from_array(np.array(1.0, dtype=np.float32), name="one"),
        numpy_helper.from_array(np.array(0.5, dtype=np.float32), name="half"),
    ]
    nodes = [
        helper.make_node("Pow", [input_name, "three"], ["cubed"]),
        helper.make_node("Mul", ["coefficient", "cubed"], ["scaled_cube"]),
        helper.make_node("Add", [input_name, "scaled_cube"], ["inner"]),
        helper.make_node("Mul", ["sqrt_2_over_pi", "inner"], ["scaled"]),
        helper.make_node("Tanh", ["scaled"], ["tanh"]),
        helper.make_node("Add", ["tanh", "one"], ["shifted"]),
        helper.make_node("Mul", [input_name, "shifted"], ["activated"]),
        helper.make_node("Mul", ["half", "activated"] if half_first else ["activated", "half"], ["y"]),
    ]
    return nodes, initializers


def _build_bias_gelu(
    *,
    approximate=None,
    shared_add=False,
    bias_first=False,
    data_shape=(1, 4, 8),
    bias_shape=(8,),
):
    bias = numpy_helper.from_array(np.ones(bias_shape, dtype=np.float32), name="bias")
    add_inputs = ["bias", "x"] if bias_first else ["x", "bias"]
    nodes = [helper.make_node("Add", add_inputs, ["add_output"])]
    gelu_attributes = {} if approximate is None else {"approximate": approximate}
    nodes.append(helper.make_node("Gelu", ["add_output"], ["y"], **gelu_attributes))
    outputs = ["y"]
    if shared_add:
        nodes.append(helper.make_node("Identity", ["add_output"], ["residual"]))
        outputs.append("residual")
    return _make_model(nodes, [bias], outputs, input_shape=data_shape)


@pytest.mark.parametrize("surgeon_type", [FuseGelu, FuseBiasGelu])
def test_activation_surgery_registers_on_module_import(surgeon_type):
    assert Surgeon.registry[surgeon_type.__name__.lower()] is surgeon_type


@pytest.mark.parametrize("half_first", [False, True])
def test_fuse_gelu_fuses_exact_variants(tmp_path, half_first):
    model = _run_surgery(_build_exact_gelu(half_first=half_first), tmp_path, "FuseGelu", f"exact_{half_first}")

    assert _count_ops(model) == {"Gelu": 1}
    gelu = model.graph.node[0]
    assert helper.get_attribute_value(next(attr for attr in gelu.attribute if attr.name == "approximate")) == b"none"


@pytest.mark.parametrize("half_first", [False, True])
def test_fuse_gelu_fuses_approximate_variants(tmp_path, half_first):
    nodes, initializers = _build_approximate_gelu(half_first=half_first)
    model = _run_surgery(_make_model(nodes, initializers, ["y"]), tmp_path, "FuseGelu", f"approx_{half_first}")

    assert _count_ops(model) == {"Gelu": 1}
    gelu = model.graph.node[0]
    assert helper.get_attribute_value(next(attr for attr in gelu.attribute if attr.name == "approximate")) == b"tanh"


@pytest.mark.parametrize(
    ("model", "remaining_op"),
    [
        (_build_exact_gelu(sqrt_2=2.0), "Erf"),
        (_build_exact_gelu(sqrt_2=[_SQRT_2, _SQRT_2]), "Erf"),
        (_make_model(*_build_approximate_gelu(coefficient=0.05), ["y"]), "Tanh"),
        (_make_model(*_build_approximate_gelu(coefficient=[0.044715, 0.044715]), ["y"]), "Tanh"),
    ],
)
def test_fuse_gelu_preserves_non_matching_constants(tmp_path, model, remaining_op):
    rewritten = _run_surgery(model, tmp_path, "FuseGelu", f"non_match_{remaining_op}")

    assert _count_ops(rewritten).get("Gelu", 0) == 0
    assert _count_ops(rewritten)[remaining_op] == 1


def test_fuse_gelu_preserves_decomposition_before_opset_20(tmp_path):
    model = _run_surgery(_build_exact_gelu(opset=13), tmp_path, "FuseGelu", "gelu_opset_13")

    assert _count_ops(model).get("Gelu", 0) == 0
    assert _count_ops(model)["Erf"] == 1


@pytest.mark.parametrize(
    ("approximate", "bias_first", "data_shape"),
    [
        (None, False, (8,)),
        ("none", False, (4, 8)),
        ("none", True, (1, 4, 8)),
        ("none", True, (1, 2, 4, 8)),
    ],
)
def test_fuse_bias_gelu_fuses_exact_gelu_and_orders_bias_last(tmp_path, approximate, bias_first, data_shape):
    model = _run_surgery(
        _build_bias_gelu(approximate=approximate, bias_first=bias_first, data_shape=data_shape),
        tmp_path,
        "FuseBiasGelu",
        f"bias_gelu_{approximate}_{bias_first}_{len(data_shape)}",
    )

    assert _count_ops(model) == {"BiasGelu": 1}
    assert model.graph.node[0].domain == "com.microsoft"
    assert list(model.graph.node[0].input) == ["x", "bias"]


def test_fuse_bias_gelu_preserves_tanh_gelu(tmp_path):
    model = _run_surgery(
        _build_bias_gelu(approximate="tanh"),
        tmp_path,
        "FuseBiasGelu",
        "tanh_bias_gelu",
    )

    assert _count_ops(model) == {"Add": 1, "Gelu": 1}


@pytest.mark.parametrize(
    ("data_shape", "bias_shape"),
    [
        ((1, 4, 8), (1, 4, 8)),
        ((1, 4, 8), (1, 8)),
        ((1, 4, 8), (7,)),
        ((), (8,)),
        ((None, None, None), (8,)),
    ],
)
def test_fuse_bias_gelu_rejects_unsafe_add_shapes(tmp_path, data_shape, bias_shape):
    model = _run_surgery(
        _build_bias_gelu(approximate="none", data_shape=data_shape, bias_shape=bias_shape),
        tmp_path,
        "FuseBiasGelu",
        f"unsafe_bias_gelu_{bias_shape}",
    )

    assert _count_ops(model) == {"Add": 1, "Gelu": 1}


def test_fuse_bias_gelu_preserves_shared_add(tmp_path):
    model = _run_surgery(
        _build_bias_gelu(approximate="none", shared_add=True),
        tmp_path,
        "FuseBiasGelu",
        "shared_bias_gelu",
    )

    assert _count_ops(model) == {"Add": 1, "Gelu": 1, "Identity": 1}


def test_fuse_bias_gelu_matches_exact_gelu_numerically(tmp_path):
    original = _build_bias_gelu(approximate="none", bias_first=True)
    rewritten = _run_surgery(original, tmp_path, "FuseBiasGelu", "bias_gelu_parity")
    inputs = {"x": np.random.default_rng(0).standard_normal((1, 4, 8), dtype=np.float32)}

    expected = InferenceSession(original.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)[0]
    actual = InferenceSession(rewritten.SerializeToString(), providers=["CPUExecutionProvider"]).run(None, inputs)[0]

    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-6)


def test_fuse_gelu_then_bias_gelu_through_public_pass(tmp_path):
    bias = numpy_helper.from_array(np.ones(8, dtype=np.float32), name="bias")
    nodes, initializers = _exact_gelu_parts(input_name="add_output")
    model = _make_model([helper.make_node("Add", ["x", "bias"], ["add_output"]), *nodes], [bias, *initializers], ["y"])
    model_path = tmp_path / "combined.onnx"
    onnx.save(model, model_path)
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {
            "surgeries": [{"surgeon": "FuseGelu"}, {"surgeon": "FuseBiasGelu"}],
            "remove_duplicate_initializers": False,
        },
        disable_search=True,
    )

    rewritten = graph_surgeries.run(
        ONNXModelHandler(model_path=str(model_path)), tmp_path / "combined_output"
    ).load_model()

    onnx.checker.check_model(rewritten)
    assert _count_ops(rewritten) == {"BiasGelu": 1}
