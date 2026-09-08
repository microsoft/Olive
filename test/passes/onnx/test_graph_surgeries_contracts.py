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


def _run_surgery(tmp_path, model, surgeon):
    input_path = tmp_path / "input.onnx"
    ir.save(model, input_path)
    p = create_pass_from_dict(
        GraphSurgeries,
        {"surgeries": [{"surgeon": surgeon}]},
        disable_search=True,
    )
    return p.run(ONNXModelHandler(model_path=str(input_path)), tmp_path / "output")


def _assert_surgery_numerics(tmp_path, output, feeds):
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
    output = _run_surgery(tmp_path, model, surgeon)
    rewritten = ir.load(output.model_path)
    assert sum(node.op_type == fused_op for node in rewritten.graph) == 1

    rng = np.random.default_rng(23)
    feeds = {
        "x": rng.standard_normal((2, 4, 8)).astype(np.float32),
        "skip": rng.standard_normal(skip_shape).astype(np.float32),
    }
    _assert_surgery_numerics(tmp_path, output, feeds)
