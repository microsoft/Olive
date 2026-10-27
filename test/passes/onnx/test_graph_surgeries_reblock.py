# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math

import numpy as np
import onnx_ir as ir
import pytest
from onnxruntime import GraphOptimizationLevel, InferenceSession, SessionOptions

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries, ReblockMatMulNBits


def _model(k, bits, explicit_zero, scale_dtype=np.float32):
    rng = np.random.default_rng(42)
    n = 8
    codes_per_byte = 8 // bits
    codes = rng.integers(0, 1 << bits, (n, math.ceil(k / codes_per_byte) * codes_per_byte), dtype=np.uint8)
    packed = np.bitwise_or.reduce(
        codes.reshape(n, -1, codes_per_byte) << (np.arange(codes_per_byte, dtype=np.uint8) * bits), axis=-1
    )
    scales = rng.uniform(0.001, 0.02, (n, 1)).astype(scale_dtype)
    zeros = (
        rng.integers(0, 1 << bits, (n, 1), dtype=np.uint8)
        if explicit_zero
        else np.full((n, 1), 1 << (bits - 1), dtype=np.uint8)
    )
    weights = (codes[:, :k].astype(np.float32) - zeros) * scales.astype(np.float32)
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 3, k])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[2, 3, n])
    initializers = [
        ir.val("weight", const_value=ir.tensor(packed.reshape(n, 1, -1))),
        ir.val("scales", const_value=ir.tensor(scales)),
    ]
    if explicit_zero:
        initializers.append(ir.val("zeros", const_value=ir.tensor(zeros)))
    node = ir.Node(
        "com.microsoft",
        "MatMulNBits",
        [x, *initializers],
        attributes=[
            ir.AttrInt64(name, value) for name, value in {"K": k, "N": n, "bits": bits, "block_size": k}.items()
        ],
        outputs=[y],
        name="head",
    )
    graph = ir.Graph([x], [y], nodes=[node], initializers=initializers, opset_imports={"": 21, "com.microsoft": 1})
    return ir.Model(graph, ir_version=10), codes[:, :k], scales, zeros, weights


def _unpack(array, bits):
    return ((array[..., None] >> (np.arange(8 // bits, dtype=np.uint8) * bits)) & ((1 << bits) - 1)).reshape(
        array.shape[0], -1
    )


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("k", [7, 33, 96, 1536])
@pytest.mark.parametrize("explicit_zero", [False, True])
@pytest.mark.parametrize("scale_dtype", [np.float16, np.float32])
def test_reblock_preserves_exact_weight_codes_and_parameters(bits, k, explicit_zero, scale_dtype):
    model, codes, scales, zeros, weights = _model(k, bits, explicit_zero, scale_dtype)
    ReblockMatMulNBits(["head"]).call_ir(model)
    node = next(iter(model.graph))
    assert node.attributes["block_size"].as_int() == 128
    count = math.ceil(k / 128)
    actual_codes = _unpack(node.inputs[1].const_value.numpy().reshape(8, -1), bits)
    np.testing.assert_array_equal(actual_codes[:, :k], codes)
    actual_scales = node.inputs[2].const_value.numpy()
    assert actual_scales.dtype == scale_dtype
    np.testing.assert_array_equal(actual_scales, np.repeat(scales, count, axis=1))
    actual_zeros = (
        _unpack(node.inputs[3].const_value.numpy(), bits)[:, :count]
        if explicit_zero
        else np.full((8, count), 1 << (bits - 1), dtype=np.uint8)
    )
    np.testing.assert_array_equal(actual_zeros, np.repeat(zeros, count, axis=1))
    columns = np.arange(k) // 128
    restored = (actual_codes[:, :k].astype(np.float32) - actual_zeros[:, columns]) * actual_scales[:, columns].astype(
        np.float32
    )
    np.testing.assert_array_equal(restored, weights)
    if k % 128:
        np.testing.assert_array_equal(actual_codes[:, k:], np.broadcast_to(zeros, actual_codes[:, k:].shape))
    first_inputs = list(node.inputs)
    ReblockMatMulNBits(["head"]).call_ir(model)
    assert list(node.inputs) == first_inputs


@pytest.mark.parametrize("bits", [4, 8])
@pytest.mark.parametrize("explicit_zero", [False, True])
def test_reblock_recipe_pass_runs_on_cpu(tmp_path, bits, explicit_zero):
    model, _, _, _, weights = _model(1536, bits, explicit_zero)
    source = tmp_path / "source.onnx"
    ir.save(model, source, external_data="source.data", size_threshold_bytes=0)
    surgery = create_pass_from_dict(
        GraphSurgeries,
        {
            "surgeries": [{"surgeon": "ReblockMatMulNBits", "node_names": ["head"], "block_size": 128}],
            "save_as_external_data": True,
        },
        disable_search=True,
    )
    output = surgery.run(ONNXModelHandler(str(source)), str(tmp_path / "reblocked"))
    options = SessionOptions()
    options.intra_op_num_threads = 1
    options.graph_optimization_level = GraphOptimizationLevel.ORT_DISABLE_ALL
    session = InferenceSession(output.model_path, options, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(12).standard_normal((2, 3, 1536)).astype(np.float32)
    expected = (x.astype(np.float64) @ weights.T.astype(np.float64)).astype(np.float32)
    np.testing.assert_allclose(session.run(None, {"x": x})[0], expected, rtol=1e-4, atol=1e-4)


def test_reblock_preserves_shared_initializers_and_unselected_node():
    model, *_ = _model(1536, 8, False)
    head = next(iter(model.graph))
    original_inputs = list(head.inputs)
    other = ir.Node(
        head.domain,
        head.op_type,
        original_inputs,
        attributes=list(head.attributes.values()),
        name="other",
    )
    model.graph.append(other)
    ReblockMatMulNBits(["head"]).call_ir(model)
    assert list(other.inputs) == original_inputs
    assert other.attributes["block_size"].as_int() == 1536
    for value in original_inputs[1:]:
        assert model.graph.initializers[value.name] is value
    assert head.inputs[1] is not other.inputs[1]


@pytest.mark.parametrize("block_size", [0, 8, 48])
def test_reblock_rejects_invalid_block_size(block_size):
    with pytest.raises(ValueError, match="power of two"):
        ReblockMatMulNBits(["head"], block_size=block_size)


def test_reblock_rejects_unknown_node_name():
    model, *_ = _model(1536, 8, False)
    with pytest.raises(ValueError, match="exactly one node"):
        ReblockMatMulNBits(["missing"]).call_ir(model)


def test_reblock_rejects_blockwise_source():
    model, *_ = _model(1536, 8, False)
    next(iter(model.graph)).attributes["block_size"] = ir.AttrInt64("block_size", 256)
    with pytest.raises(ValueError, match="single-block"):
        ReblockMatMulNBits(["head"]).call_ir(model)


def test_reblock_rejects_gidx():
    model, *_ = _model(1536, 8, False)
    node = next(iter(model.graph))
    node.resize_inputs(5)
    indices = ir.val("g_idx", const_value=ir.tensor(np.zeros(1536, dtype=np.int32)))
    model.graph.register_initializer(indices)
    node.replace_input_with(4, indices)
    with pytest.raises(ValueError, match="g_idx"):
        ReblockMatMulNBits(["head"]).call_ir(model)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_reblock_rejects_dynamic_parameters(index):
    model, *_ = _model(1536, 8, True)
    next(iter(model.graph)).inputs[index].const_value = None
    with pytest.raises(ValueError, match="must be initializers"):
        ReblockMatMulNBits(["head"]).call_ir(model)


@pytest.mark.parametrize("index", [1, 2, 3])
def test_reblock_rejects_malformed_parameters(index):
    model, *_ = _model(1536, 8, True)
    node = next(iter(model.graph))
    node.inputs[index].const_value = ir.tensor(np.zeros((1,), dtype=np.uint8))
    with pytest.raises(ValueError, match=r"invalid packed|one scale|one packed"):
        ReblockMatMulNBits(["head"]).call_ir(model)


def test_reblock_preserves_optional_bias_and_empty_gidx():
    model, *_ = _model(1536, 8, False)
    node = next(iter(model.graph))
    node.resize_inputs(6)
    bias = ir.val("bias", const_value=ir.tensor(np.arange(8, dtype=np.float32)))
    model.graph.register_initializer(bias)
    node.replace_input_with(5, bias)
    ReblockMatMulNBits(["head"]).call_ir(model)
    assert node.inputs[3] is None
    assert node.inputs[4] is None
    assert node.inputs[5] is bias
