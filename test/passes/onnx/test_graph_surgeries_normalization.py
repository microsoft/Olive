# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import numpy as np
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.passes.onnx.graph_surgery.base import Surgeon
from olive.passes.onnx.graph_surgery.normalization import (
    FuseLayerNormalization,
    FuseSkipLayerNormalization,
    FuseSkipRMSNormalization,
)
from test.passes.onnx.graph_surgery_test_utils import count_ops as _count_ops
from test.passes.onnx.graph_surgery_test_utils import run_surgery as _run_surgery


def _build_decomposed_layer_normalization(*, include_bias=True, axes=-1, exponent=2.0, epsilon=1e-5):
    x = helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 4, 8])
    y = helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 4, 8])
    initializers = [
        numpy_helper.from_array(np.array([axes], dtype=np.int64), name="axes"),
        numpy_helper.from_array(np.array(exponent, dtype=np.float32), name="exponent"),
        numpy_helper.from_array(np.array(epsilon, dtype=np.float32), name="epsilon"),
        numpy_helper.from_array(np.ones(8, dtype=np.float32), name="weight"),
    ]
    nodes = [
        helper.make_node("ReduceMean", ["x", "axes"], ["mean"], keepdims=1),
        helper.make_node("Sub", ["x", "mean"], ["difference"]),
        helper.make_node("Pow", ["difference", "exponent"], ["squared"]),
        helper.make_node("ReduceMean", ["squared", "axes"], ["variance"], keepdims=1),
        helper.make_node("Add", ["variance", "epsilon"], ["variance_epsilon"]),
        helper.make_node("Sqrt", ["variance_epsilon"], ["standard_deviation"]),
        helper.make_node("Div", ["difference", "standard_deviation"], ["normalized"]),
        helper.make_node("Mul", ["normalized", "weight"], ["scaled" if include_bias else "y"]),
    ]
    if include_bias:
        initializers.append(numpy_helper.from_array(np.zeros(8, dtype=np.float32), name="bias"))
        nodes.append(helper.make_node("Add", ["scaled", "bias"], ["y"]))
    graph = helper.make_graph(nodes, "layer_normalization_test", [x], [y], initializers)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 21)])


def _build_skip_normalization(
    norm_op_type,
    *,
    include_bias=False,
    shared_add=False,
    epsilon=1e-5,
    axis=-1,
    input_shape=(1, 4, 8),
    skip_shape=None,
    unknown_skip_shape=False,
    skip_first=False,
    dtype=TensorProto.FLOAT,
    add_is_graph_output=False,
):
    x = helper.make_tensor_value_info("x", dtype, input_shape)
    effective_skip_shape = (
        (None,) * len(input_shape) if unknown_skip_shape else (input_shape if skip_shape is None else skip_shape)
    )
    skip = helper.make_tensor_value_info("skip", dtype, effective_skip_shape)
    y = helper.make_tensor_value_info("y", dtype, input_shape)
    numpy_dtype = np.float64 if dtype == TensorProto.DOUBLE else np.float32
    initializers = [numpy_helper.from_array(np.ones(8, dtype=numpy_dtype), name="weight")]
    add_inputs = ["skip", "x"] if skip_first else ["x", "skip"]
    nodes = [helper.make_node("Add", add_inputs, ["add_output"])]
    norm_inputs = ["add_output", "weight"]
    if include_bias:
        initializers.append(numpy_helper.from_array(np.zeros(8, dtype=numpy_dtype), name="bias"))
        norm_inputs.append("bias")
    attributes = {} if epsilon is None else {"epsilon": epsilon}
    if axis is not None:
        attributes["axis"] = axis
    nodes.append(helper.make_node(norm_op_type, norm_inputs, ["y"], **attributes))

    outputs = [y]
    if shared_add:
        nodes.append(helper.make_node("Identity", ["add_output"], ["residual"]))
        outputs.append(helper.make_tensor_value_info("residual", dtype, input_shape))
    if add_is_graph_output:
        outputs.append(helper.make_tensor_value_info("add_output", dtype, input_shape))

    graph = helper.make_graph(nodes, "skip_normalization_test", [x, skip], outputs, initializers)
    return helper.make_model(graph, ir_version=10, opset_imports=[helper.make_opsetid("", 23)])


@pytest.mark.parametrize(
    "surgeon_type",
    [FuseLayerNormalization, FuseSkipLayerNormalization, FuseSkipRMSNormalization],
)
def test_normalization_surgery_registers_on_module_import(surgeon_type):
    assert Surgeon.registry[surgeon_type.__name__.lower()] is surgeon_type


@pytest.mark.parametrize("include_bias", [False, True])
def test_fuse_layer_normalization_fuses_bias_variants(tmp_path, include_bias):
    model = _run_surgery(
        _build_decomposed_layer_normalization(include_bias=include_bias),
        tmp_path,
        "FuseLayerNormalization",
        f"layer_norm_{include_bias}",
    )

    assert _count_ops(model) == {"LayerNormalization": 1}
    layer_norm = model.graph.node[0]
    assert helper.get_attribute_value(next(attr for attr in layer_norm.attribute if attr.name == "axis")) == -1
    assert helper.get_attribute_value(
        next(attr for attr in layer_norm.attribute if attr.name == "epsilon")
    ) == pytest.approx(1e-5)
    assert len(layer_norm.input) == (3 if include_bias else 2)


@pytest.mark.parametrize(
    ("kwargs", "remaining_op"),
    [
        ({"axes": -2}, "ReduceMean"),
        ({"exponent": 3.0}, "Pow"),
        ({"exponent": [2.0, 2.0]}, "Pow"),
        ({"epsilon": 2.0}, "Sqrt"),
        ({"epsilon": [1e-5, 1e-5]}, "Sqrt"),
    ],
)
def test_fuse_layer_normalization_preserves_non_matches(tmp_path, kwargs, remaining_op):
    model = _run_surgery(
        _build_decomposed_layer_normalization(**kwargs),
        tmp_path,
        "FuseLayerNormalization",
        f"layer_norm_non_match_{remaining_op}",
    )

    counts = _count_ops(model)
    assert counts.get("LayerNormalization", 0) == 0
    assert counts[remaining_op] >= 1


@pytest.mark.parametrize(("include_bias", "input_shape"), [(False, (4, 8)), (True, (1, 4, 8))])
def test_fuse_skip_layer_normalization_rewires_shared_residual(tmp_path, include_bias, input_shape):
    model = _run_surgery(
        _build_skip_normalization(
            "LayerNormalization",
            include_bias=include_bias,
            shared_add=True,
            input_shape=input_shape,
        ),
        tmp_path,
        "FuseSkipLayerNormalization",
        f"skip_layer_norm_{include_bias}_{len(input_shape)}",
    )

    assert _count_ops(model) == {"Identity": 1, "SkipLayerNormalization": 1}
    fused = next(node for node in model.graph.node if node.op_type == "SkipLayerNormalization")
    residual = next(node for node in model.graph.node if node.op_type == "Identity")
    assert fused.domain == "com.microsoft"
    assert residual.input[0] == fused.output[3]
    assert len(fused.input) == (4 if include_bias else 3)


def test_fuse_skip_layer_normalization_fuses_single_consumer_add(tmp_path):
    model = _run_surgery(
        _build_skip_normalization("LayerNormalization", include_bias=True),
        tmp_path,
        "FuseSkipLayerNormalization",
        "skip_layer_norm_single_consumer",
    )

    assert _count_ops(model) == {"SkipLayerNormalization": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": 0},
        {"input_shape": (1, 1, 4, 8)},
        {"skip_shape": (8,)},
        {"skip_shape": (5, 8)},
        {"skip_shape": (1, 2, 8)},
        {"input_shape": (2, 4, 8), "skip_shape": (3, 4, 8)},
        {"unknown_skip_shape": True},
        {"add_is_graph_output": True},
    ],
)
def test_fuse_skip_layer_normalization_preserves_non_matches(tmp_path, kwargs):
    model = _run_surgery(
        _build_skip_normalization("LayerNormalization", include_bias=True, **kwargs),
        tmp_path,
        "FuseSkipLayerNormalization",
        f"skip_layer_norm_non_match_{next(iter(kwargs))}",
    )

    counts = _count_ops(model)
    assert counts.get("SkipLayerNormalization", 0) == 0
    assert counts["Add"] == 1
    assert counts["LayerNormalization"] == 1


@pytest.mark.parametrize(
    ("norm_op_type", "surgeon", "fused_op"),
    [
        ("LayerNormalization", "FuseSkipLayerNormalization", "SkipLayerNormalization"),
        ("RMSNormalization", "FuseSkipRMSNormalization", "SkipSimplifiedLayerNormalization"),
    ],
)
@pytest.mark.parametrize("skip_shape", [(4, 8), (1, 4, 8)])
def test_fuse_skip_normalization_supports_ort_skip_broadcast_and_reorders_data(
    tmp_path, norm_op_type, surgeon, fused_op, skip_shape
):
    model = _run_surgery(
        _build_skip_normalization(
            norm_op_type,
            input_shape=(2, 4, 8),
            skip_shape=skip_shape,
            skip_first=True,
        ),
        tmp_path,
        surgeon,
        f"broadcast_{norm_op_type}_{len(skip_shape)}",
    )

    assert _count_ops(model) == {fused_op: 1}
    assert list(model.graph.node[0].input[:2]) == ["x", "skip"]


@pytest.mark.parametrize(
    ("norm_op_type", "surgeon", "fused_op"),
    [
        ("LayerNormalization", "FuseSkipLayerNormalization", "SkipLayerNormalization"),
        ("RMSNormalization", "FuseSkipRMSNormalization", "SkipSimplifiedLayerNormalization"),
    ],
)
def test_fuse_skip_normalization_emits_default_epsilon(tmp_path, norm_op_type, surgeon, fused_op):
    model = _run_surgery(
        _build_skip_normalization(norm_op_type, epsilon=None, axis=None),
        tmp_path,
        surgeon,
        f"default_epsilon_{norm_op_type}",
    )

    assert _count_ops(model) == {fused_op: 1}
    fused = model.graph.node[0]
    epsilon = helper.get_attribute_value(next(attr for attr in fused.attribute if attr.name == "epsilon"))
    assert epsilon == pytest.approx(1e-5)


@pytest.mark.parametrize(
    ("norm_op_type", "surgeon", "fused_op", "include_bias"),
    [
        ("LayerNormalization", "FuseSkipLayerNormalization", "SkipLayerNormalization", True),
        ("RMSNormalization", "FuseSkipRMSNormalization", "SkipSimplifiedLayerNormalization", False),
    ],
)
def test_fuse_skip_normalization_preserves_double_norm(tmp_path, norm_op_type, surgeon, fused_op, include_bias):
    model = _run_surgery(
        _build_skip_normalization(norm_op_type, include_bias=include_bias, dtype=TensorProto.DOUBLE),
        tmp_path,
        surgeon,
        f"double_{norm_op_type}",
    )

    counts = _count_ops(model)
    assert counts.get(fused_op, 0) == 0
    assert counts["Add"] == 1
    assert counts[norm_op_type] == 1


def test_fuse_skip_rms_normalization_rewires_shared_residual(tmp_path):
    model = _run_surgery(
        _build_skip_normalization("RMSNormalization", shared_add=True),
        tmp_path,
        "FuseSkipRMSNormalization",
        "skip_rms_norm_shared",
    )

    assert _count_ops(model) == {"Identity": 1, "SkipSimplifiedLayerNormalization": 1}
    fused = next(node for node in model.graph.node if node.op_type == "SkipSimplifiedLayerNormalization")
    residual = next(node for node in model.graph.node if node.op_type == "Identity")
    assert fused.domain == "com.microsoft"
    assert residual.input[0] == fused.output[3]


def test_fuse_skip_rms_normalization_fuses_single_consumer_add(tmp_path):
    model = _run_surgery(
        _build_skip_normalization("RMSNormalization"),
        tmp_path,
        "FuseSkipRMSNormalization",
        "skip_rms_norm_single_consumer",
    )

    assert _count_ops(model) == {"SkipSimplifiedLayerNormalization": 1}


@pytest.mark.parametrize(
    "kwargs",
    [
        {"axis": 0},
        {"input_shape": (1, 1, 4, 8)},
        {"skip_shape": (8,)},
        {"skip_shape": (5, 8)},
        {"skip_shape": (1, 2, 8)},
        {"input_shape": (2, 4, 8), "skip_shape": (3, 4, 8)},
        {"unknown_skip_shape": True},
        {"add_is_graph_output": True},
    ],
)
def test_fuse_skip_rms_normalization_preserves_non_matches(tmp_path, kwargs):
    model = _run_surgery(
        _build_skip_normalization("RMSNormalization", **kwargs),
        tmp_path,
        "FuseSkipRMSNormalization",
        f"skip_rms_norm_non_match_{next(iter(kwargs))}",
    )

    counts = _count_ops(model)
    assert counts.get("SkipSimplifiedLayerNormalization", 0) == 0
    assert counts["Add"] == 1
    assert counts["RMSNormalization"] == 1
