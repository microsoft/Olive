# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx_ir as ir
import pytest
from onnxruntime import GraphOptimizationLevel as OptLevel
from onnxruntime import InferenceSession, SessionOptions

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries

_MIGRATED_SURGEONS = (
    "FuseGelu",
    "FuseBiasGelu",
    "FuseLayerNormalization",
    "FuseSkipLayerNormalization",
    "FuseSkipRMSNormalization",
    "AttentionToGroupQueryAttention",
    "PackQKVForGroupQueryAttention",
    "SeparateGroupQueryAttentionRoPE",
    "UnpackGroupQueryAttentionQKV",
    "BlockDiagonalAttentionToPackedMHA",
    "ClipToMinMax",
    "Rank4RMSNormToRank3",
    "DecomposeOnnxRotaryEmbedding",
    "TensorScatterToScatterND",
    "DecomposeAttention",
    "StaticEmptyKV",
    "FuseDenseMoEToQMoE",
    "FuseBlockQuantizedMoE",
)


def test_graph_surgeries_registers_builtins_without_exporter_imports():
    # A fresh interpreter prevents test collection from masking missing production imports.
    script = """
import sys
from importlib.abc import MetaPathFinder

class BlockExporterImports(MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "mobius" or fullname.startswith("mobius."):
            raise ImportError("Graph surgeries must not import Mobius")
        return None

sys.meta_path.insert(0, BlockExporterImports())
from olive.passes.onnx.graph_surgeries import GraphSurgeries, Surgeon, RewriteRuleSurgeon, ProtoSurgeon
from olive.passes.onnx.graph_surgery.base import Surgeon as BaseSurgeon
from olive.passes.olive_pass import create_pass_from_dict

assert Surgeon is BaseSurgeon
p = create_pass_from_dict(GraphSurgeries, {"surgeries": []}, disable_search=True)
for name in sys.argv[1:]:
    instance = p.init_surgeon_instance({"surgeon": name.swapcase()})
    assert isinstance(instance, Surgeon), name
    assert type(instance).__name__ == name
assert isinstance(p.init_surgeon_instance({"surgeon": "ReplaceErfWithTanh"}), RewriteRuleSurgeon)
assert isinstance(p.init_surgeon_instance({"surgeon": "DeduplicateNodes"}), ProtoSurgeon)
assert type(p.init_surgeon_instance({"surgeon": "DecomposeRotaryEmbedding"})).__module__.endswith("graph_surgeries")
assert not any(name == "mobius" or name.startswith("mobius.") for name in sys.modules)
"""
    result = subprocess.run(
        [sys.executable, "-c", script, *_MIGRATED_SURGEONS],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _run_pipeline(tmp_path, model, surgeons):
    input_path = tmp_path / "input.onnx"
    ir.save(model, input_path)
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": surgeon} for surgeon in surgeons]},
        disable_search=True,
    )
    return p.run(ONNXModelHandler(model_path=str(input_path)), tmp_path / "output")


def _assert_pipeline_numerics(tmp_path, output, feeds):
    options = SessionOptions()
    options.graph_optimization_level = OptLevel.ORT_DISABLE_ALL
    expected = InferenceSession(
        str(tmp_path / "input.onnx"), sess_options=options, providers=["CPUExecutionProvider"]
    ).run(None, feeds)
    actual = InferenceSession(output.model_path, sess_options=options, providers=["CPUExecutionProvider"]).run(
        None, feeds
    )
    for actual_output, expected_output in zip(actual, expected):
        np.testing.assert_allclose(actual_output, expected_output, rtol=1e-6, atol=1e-6)


def test_graph_surgeries_preserves_nonmatching_model_and_metadata(tmp_path):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=["batch", 8])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=["batch", 8])
    model = ir.Model(
        ir.Graph([x], [y], nodes=[ir.Node("", "Relu", [x], outputs=[y])], opset_imports={"": 24}),
        ir_version=10,
    )
    model.metadata_props["pipeline.component"] = "encoder"
    y.metadata_props["output.role"] = "hidden_states"

    output = ir.load(_run_pipeline(tmp_path, model, _MIGRATED_SURGEONS).model_path)

    assert [(node.domain, node.op_type) for node in output.graph] == [("", "Relu")]
    assert output.metadata_props["pipeline.component"] == "encoder"
    assert output.graph.outputs[0].metadata_props["output.role"] == "hidden_states"
    assert [value.name for value in output.graph.inputs] == ["x"]
    assert [value.name for value in output.graph.outputs] == ["y"]
    assert output.graph.outputs[0].shape == y.shape
    assert output.graph.outputs[0].dtype == y.dtype


def _exact_gelu_with_bias():
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 8])
    bias = ir.val("bias", const_value=ir.tensor(np.linspace(-0.5, 0.5, 8, dtype=np.float32)))
    divisor = ir.val("divisor", const_value=ir.tensor(np.array(np.sqrt(2), dtype=np.float32)))
    one = ir.val("one", const_value=ir.tensor(np.array(1, dtype=np.float32)))
    half = ir.val("half", const_value=ir.tensor(np.array(0.5, dtype=np.float32)))
    add = ir.Node("", "Add", [x, bias])
    add.outputs[0].shape = x.shape
    add.outputs[0].dtype = x.dtype
    div = ir.Node("", "Div", [add.outputs[0], divisor])
    erf = ir.Node("", "Erf", [div.outputs[0]])
    shifted = ir.Node("", "Add", [erf.outputs[0], one])
    scaled = ir.Node("", "Mul", [add.outputs[0], shifted.outputs[0]])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=[2, 8])
    result = ir.Node("", "Mul", [scaled.outputs[0], half], outputs=[y])
    return ir.Model(
        ir.Graph(
            [x],
            [y],
            nodes=[add, div, erf, shifted, scaled, result],
            initializers=[bias, divisor, one, half],
            opset_imports={"": 21},
        ),
        ir_version=10,
    )


@pytest.mark.parametrize(
    ("surgeons", "expected_ops"),
    [
        (["FuseGelu", "FuseBiasGelu"], ["BiasGelu"]),
        (["FuseBiasGelu", "FuseGelu"], ["Add", "Gelu"]),
    ],
)
def test_graph_surgeries_respects_fusion_order_and_preserves_numerics(tmp_path, surgeons, expected_ops):
    output = _run_pipeline(tmp_path, _exact_gelu_with_bias(), surgeons)
    model = ir.load(output.model_path)
    assert [node.op_type for node in model.graph] == expected_ops
    assert model.graph.outputs[0].name == "y"
    feeds = {"x": np.random.default_rng(42).standard_normal((2, 8)).astype(np.float32)}
    _assert_pipeline_numerics(tmp_path, output, feeds)


def _gqa_projections(dtype, with_bias):
    hidden = ir.val("hidden", dtype=dtype, shape=[1, 2, 8])
    past_key = ir.val("past_key", dtype=dtype, shape=[1, 2, 6, 8])
    past_value = ir.val("past_value", dtype=dtype, shape=[1, 2, 6, 8])
    lengths = ir.val("seqlens", dtype=ir.DataType.INT32, shape=[1])
    total = ir.val("total", dtype=ir.DataType.INT32, shape=[])
    outputs = [
        ir.val("output", dtype=dtype, shape=[1, 2, 32]),
        ir.val("present_key", dtype=dtype, shape=[1, 2, 6, 8]),
        ir.val("present_value", dtype=dtype, shape=[1, 2, 6, 8]),
    ]
    rng = np.random.default_rng(7)
    nodes, initializers, projections = [], [], []
    for name, width in (("q", 32), ("k", 16), ("v", 16)):
        array = (rng.standard_normal((width, 8)) * 0.1).astype(dtype.numpy())
        weight = ir.val(f"{name}_weight", const_value=ir.tensor(array))
        initializers.append(weight)
        transpose = ir.Node("", "Transpose", [weight], attributes=[ir.AttrInt64s("perm", [1, 0])])
        matmul = ir.Node("", "MatMul", [hidden, transpose.outputs[0]])
        nodes.extend([transpose, matmul])
        projection = matmul.outputs[0]
        if with_bias:
            bias = ir.val(f"{name}_bias", const_value=ir.tensor(rng.standard_normal(width).astype(dtype.numpy())))
            initializers.append(bias)
            add = ir.Node("", "Add", [projection, bias])
            nodes.append(add)
            projection = add.outputs[0]
        projections.append(projection)
    nodes.append(
        ir.Node(
            "com.microsoft",
            "GroupQueryAttention",
            [*projections, past_key, past_value, lengths, total, None, None],
            outputs=outputs,
            attributes=[
                ir.AttrInt64("num_heads", 4),
                ir.AttrInt64("kv_num_heads", 2),
                ir.AttrInt64("do_rotary", 0),
                ir.AttrFloat32("scale", 0.5),
            ],
        )
    )
    return ir.Model(
        ir.Graph(
            [hidden, past_key, past_value, lengths, total],
            outputs,
            nodes=nodes,
            initializers=initializers,
            opset_imports={"": 24, "com.microsoft": 1},
        ),
        ir_version=10,
    )


@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16])
@pytest.mark.parametrize("with_bias", [False, True])
def test_graph_surgeries_pack_unpack_preserves_weight_values_and_cache_contract(tmp_path, dtype, with_bias):
    original = _gqa_projections(dtype, with_bias)
    output = _run_pipeline(tmp_path, original, ["PackQKVForGroupQueryAttention", "UnpackGroupQueryAttentionQKV"])
    rewritten = ir.load(output.model_path)
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")
    assert [value.name if value is not None else None for value in gqa.inputs[3:]] == [
        "past_key",
        "past_value",
        "seqlens",
        "total",
        None,
        None,
    ]
    for actual, expected in zip(rewritten.graph.outputs, original.graph.outputs):
        assert (actual.name, actual.dtype, actual.shape) == (expected.name, expected.dtype, expected.shape)
    for index, name in enumerate(("q", "k", "v")):
        projection = gqa.inputs[index].producer()
        if with_bias:
            assert projection.op_type == "Add"
            actual_bias = projection.inputs[1].const_value
            expected_bias = original.graph.initializers[f"{name}_bias"].const_value
            assert actual_bias.tobytes() == expected_bias.tobytes()
            projection = projection.inputs[0].producer()
        assert projection.op_type == "MatMul"
        weight = projection.inputs[1].producer().inputs[0]
        assert weight.dtype == dtype
        assert weight.const_value.tobytes() == original.graph.initializers[f"{name}_weight"].const_value.tobytes()

    if dtype == ir.DataType.FLOAT:
        rng = np.random.default_rng(17)
        feeds = {
            "hidden": rng.standard_normal((1, 2, 8)).astype(np.float32),
            "past_key": rng.standard_normal((1, 2, 6, 8)).astype(np.float32),
            "past_value": rng.standard_normal((1, 2, 6, 8)).astype(np.float32),
            "seqlens": np.array([5], dtype=np.int32),
            "total": np.array(6, dtype=np.int32),
        }
        _assert_pipeline_numerics(tmp_path, output, feeds)


@pytest.mark.parametrize(
    ("norm_op", "surgeon", "fused_op"),
    [
        ("LayerNormalization", "FuseSkipLayerNormalization", "SkipLayerNormalization"),
        ("RMSNormalization", "FuseSkipRMSNormalization", "SkipSimplifiedLayerNormalization"),
    ],
)
@pytest.mark.parametrize("skip_shape", [(4, 8), (1, 4, 8)])
def test_graph_surgeries_skip_broadcast_preserves_numerics_and_residual(
    tmp_path, norm_op, surgeon, fused_op, skip_shape
):
    x = ir.val("x", dtype=ir.DataType.FLOAT, shape=[2, 4, 8])
    skip = ir.val("skip", dtype=ir.DataType.FLOAT, shape=skip_shape)
    weight = ir.val("weight", const_value=ir.tensor(np.linspace(0.5, 1.5, 8, dtype=np.float32)))
    add = ir.Node("", "Add", [skip, x])
    y = ir.val("y", dtype=ir.DataType.FLOAT, shape=x.shape)
    residual = ir.val("residual", dtype=ir.DataType.FLOAT, shape=x.shape)
    model = ir.Model(
        ir.Graph(
            [x, skip],
            [y, residual],
            nodes=[
                add,
                ir.Node("", norm_op, [add.outputs[0], weight], outputs=[y]),
                ir.Node("", "Identity", [add.outputs[0]], outputs=[residual]),
            ],
            initializers=[weight],
            opset_imports={"": 24},
        ),
        ir_version=10,
    )
    output = _run_pipeline(tmp_path, model, [surgeon])
    rewritten = ir.load(output.model_path)
    assert sum(node.op_type == fused_op for node in rewritten.graph) == 1

    rng = np.random.default_rng(23)
    feeds = {
        "x": rng.standard_normal((2, 4, 8)).astype(np.float32),
        "skip": rng.standard_normal(skip_shape).astype(np.float32),
    }
    _assert_pipeline_numerics(tmp_path, output, feeds)
