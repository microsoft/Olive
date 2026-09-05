# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Tests for OnnxMoEQuantization (com.microsoft::MoE → com.microsoft::QMoE)."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.moe_quantization import OnnxMoEQuantization


def _build_moe_model(tmp_path, num_experts=4, hidden=16, inter=32, top_k=2):
    """Build a tiny ONNX model containing one ``com.microsoft::MoE`` node.

    Layout matches what mobius emits for Gemma 4 MoE:
        fc1_experts_weights  [E, 2*inter, H]
        fc2_experts_weights  [E, H, inter]
    """
    rng = np.random.RandomState(0)
    fc1 = rng.randn(num_experts, 2 * inter, hidden).astype(np.float32) * 0.02
    fc2 = rng.randn(num_experts, hidden, inter).astype(np.float32) * 0.02

    fc1_init = numpy_helper.from_array(fc1, name="fc1_W")
    fc2_init = numpy_helper.from_array(fc2, name="fc2_W")

    input_t = helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, hidden])
    router_t = helper.make_tensor_value_info("router", TensorProto.FLOAT, [None, num_experts])
    output_t = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, hidden])

    moe = helper.make_node(
        "MoE",
        inputs=["x", "router", "fc1_W", "", "fc2_W"],
        outputs=["y"],
        name="moe_layer_0",
        domain="com.microsoft",
        k=top_k,
        normalize_routing_weights=1,
        activation_type="swiglu",
        swiglu_fusion=1,
        activation_alpha=1.0,
        activation_beta=0.0,
        swiglu_limit=float("inf"),
    )

    graph = helper.make_graph(
        nodes=[moe],
        name="moe_only",
        inputs=[input_t, router_t],
        outputs=[output_t],
        initializer=[fc1_init, fc2_init],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 20),
            helper.make_opsetid("com.microsoft", 1),
        ],
    )
    model.ir_version = 10
    path = tmp_path / "moe.onnx"
    onnx.save(model, path)
    return path, fc1, fc2


def _fake_pack_weights_for_cuda_mixed_gemm(q_weights, n, k, bits, force_arch):
    """Pass-through replacement for the CUTLASS prepack helper.

    The real helper permutes / interleaves bytes for the fpA_intB kernel
    and is only available when ORT is built with USE_CUDA. The structural
    pass test doesn't depend on the byte layout; this stub keeps the
    shape (``[k, n // pack_factor]`` after the caller's reshape) but
    leaves the data identical so the test runs in CPU-only CI.
    """
    pack_factor = 8 // bits
    out = np.ascontiguousarray(q_weights.reshape(n, k // pack_factor))
    # Caller reshapes to (k, n // pack_factor) — return the same total byte
    # count so that reshape doesn't fail.
    return out.reshape(k, n // pack_factor)


def test_moe_to_qmoe_conversion(tmp_path):
    """Replace one MoE node with a QMoE node, int4-quantize, carry attrs.

    End-to-end: a single MoE node is replaced by a QMoE node, weights are
    quantized to int4, scales are added, and all routing/activation attrs
    are carried over.
    """
    model_path, fc1_fp32, fc2_fp32 = _build_moe_model(tmp_path)

    input_model = ONNXModelHandler(model_path=str(model_path))
    p = create_pass_from_dict(
        OnnxMoEQuantization,
        {"bits": 4, "block_size": 0},
        disable_search=True,
    )

    with patch(
        "olive.passes.onnx.moe_quantization._load_cuda_pack_fn",
        return_value=_fake_pack_weights_for_cuda_mixed_gemm,
    ):
        output_model = p.run(input_model, str(tmp_path / "out"))

    g = output_model.load_model().graph

    moe_nodes = [n for n in g.node if n.op_type == "MoE"]
    qmoe_nodes = [n for n in g.node if n.op_type == "QMoE"]
    assert moe_nodes == [], "Original MoE node should be replaced."
    assert len(qmoe_nodes) == 1
    qmoe = qmoe_nodes[0]
    assert qmoe.domain == "com.microsoft"
    # QMoE input ordering: input, router, fc1_W, fc1_scales, fc1_zp, fc1_b,
    # fc2_W, fc2_scales, fc2_zp, fc2_b
    assert qmoe.input[0] == "x"
    assert qmoe.input[1] == "router"
    assert qmoe.input[2].endswith("_q"), "Quantized weight initializer expected at slot 2."
    assert qmoe.input[3].endswith("_scales"), "Scale initializer expected at slot 3."
    assert qmoe.input[4] == "", "Zero-point (symmetric mode) should be empty at slot 4."
    assert qmoe.input[6].endswith("_q")
    assert qmoe.input[7].endswith("_scales")

    # Attributes: routing/activation carried over plus expert_weight_bits.
    attrs = {a.name: a for a in qmoe.attribute}
    assert attrs["k"].i == 2
    assert attrs["normalize_routing_weights"].i == 1
    assert attrs["activation_type"].s.decode() == "swiglu"
    assert attrs["swiglu_fusion"].i == 1
    assert attrs["expert_weight_bits"].i == 4
    assert "block_size" not in attrs, "block_size should not be emitted when 0."
    assert attrs["quant_type"].s.decode() == "int"

    # Initializer dtype + shape checks.
    inits = {i.name: i for i in g.initializer}
    fc1_q = numpy_helper.to_array(inits[qmoe.input[2]])
    fc1_s = numpy_helper.to_array(inits[qmoe.input[3]])
    fc2_q = numpy_helper.to_array(inits[qmoe.input[6]])
    fc2_s = numpy_helper.to_array(inits[qmoe.input[7]])

    e_dim, two_inter, hidden = fc1_fp32.shape  # [E, 2*inter, H]
    pack_factor = 2  # int4
    assert fc1_q.dtype == np.uint8
    assert fc1_q.shape == (e_dim, hidden, two_inter // pack_factor)
    assert fc1_s.dtype == np.float16
    assert fc1_s.shape == (e_dim, two_inter)  # per-row scales when block_size == 0

    e_dim2, hidden2, inter = fc2_fp32.shape  # [E, H, inter]
    assert fc2_q.dtype == np.uint8
    assert fc2_q.shape == (e_dim2, inter, hidden2 // pack_factor)
    assert fc2_s.shape == (e_dim2, hidden2)

    # The original fp32 weights must be gone.
    assert "fc1_W" not in inits
    assert "fc2_W" not in inits


def test_moe_to_qmoe_blockwise(tmp_path):
    """Block-wise (block_size=16) emits 3-D scales and a block_size attribute."""
    model_path, fc1, _ = _build_moe_model(tmp_path, hidden=32, inter=32)
    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4, "block_size": 16}, disable_search=True)
    input_model = ONNXModelHandler(model_path=str(model_path))
    with patch(
        "olive.passes.onnx.moe_quantization._load_cuda_pack_fn",
        return_value=_fake_pack_weights_for_cuda_mixed_gemm,
    ):
        output_model = p.run(input_model, str(tmp_path / "out"))
    g = output_model.load_model().graph
    qmoe = next(n for n in g.node if n.op_type == "QMoE")
    attrs = {a.name: a for a in qmoe.attribute}
    assert attrs["block_size"].i == 16

    inits = {i.name: i for i in g.initializer}
    fc1_s = numpy_helper.to_array(inits[qmoe.input[3]])
    e_dim, two_inter, hidden = fc1.shape
    # Block-wise scales: [E, N, K // block_size]
    assert fc1_s.shape == (e_dim, two_inter, hidden // 16)


def test_moe_to_qmoe_skip_when_not_initializer(tmp_path):
    """Skip an MoE node whose weight is a dynamic input, leaving it unchanged."""
    model_path, _, _ = _build_moe_model(tmp_path)
    # Edit the model: replace fc1_W initializer with a graph input so it
    # isn't a static initializer.
    m = onnx.load(model_path)
    m.graph.initializer.pop(0)  # remove fc1_W
    m.graph.input.append(helper.make_tensor_value_info("fc1_W", TensorProto.FLOAT, [4, 64, 16]))
    onnx.save(m, model_path)

    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4}, disable_search=True)
    input_model = ONNXModelHandler(model_path=str(model_path))
    with patch(
        "olive.passes.onnx.moe_quantization._load_cuda_pack_fn",
        return_value=_fake_pack_weights_for_cuda_mixed_gemm,
    ):
        output_model = p.run(input_model, str(tmp_path / "out"))
    g = output_model.load_model().graph
    assert [n.op_type for n in g.node] == ["MoE"], "Node should remain unchanged when weights are dynamic."


def test_invalid_bits_rejected(tmp_path):
    """Bits other than 4 or 8 fails fast at config time."""
    model_path, _, _ = _build_moe_model(tmp_path)
    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 5}, disable_search=True)
    with pytest.raises(ValueError, match="bits must be 4 or 8"):
        p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "out"))


def test_invalid_block_size_rejected(tmp_path):
    """Non-power-of-two block_size fails fast."""
    model_path, _, _ = _build_moe_model(tmp_path)
    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4, "block_size": 24}, disable_search=True)
    with pytest.raises(ValueError, match="power of two"):
        p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "out"))


def test_moe_to_qmoe_handles_explicit_empty_optional_inputs(tmp_path):
    """Convert an MoE node with explicit empty-string optional inputs.

    Optional fc1_bias, fc3_W, and fc3_bias slots are present as empty
    strings rather than absent slots; the pass should treat them as
    missing and still convert the node.
    """
    # _build_moe_model already emits inputs=['x','router','fc1_W','','fc2_W']; here we
    # extend it to also include empty fc3 slots to exercise the slot-7 boundary.
    model_path, _, _ = _build_moe_model(tmp_path)
    m = onnx.load(model_path)
    moe = m.graph.node[0]
    # Append empty fc2_bias, fc3_W, fc3_bias slots.
    moe.input.extend(["", "", ""])
    onnx.save(m, model_path)

    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4}, disable_search=True)
    with patch(
        "olive.passes.onnx.moe_quantization._load_cuda_pack_fn",
        return_value=_fake_pack_weights_for_cuda_mixed_gemm,
    ):
        output_model = p.run(ONNXModelHandler(model_path=str(model_path)), str(tmp_path / "out"))
    g = output_model.load_model().graph
    qmoe_nodes = [n for n in g.node if n.op_type == "QMoE"]
    assert len(qmoe_nodes) == 1, "MoE node with empty optional slots should still be converted."


def test_n_not_divisible_by_pack_factor_skipped(tmp_path):
    """Skip MoE nodes whose N is incompatible with the 4-bit packing factor.

    N (== 2*inter for fc1) not divisible by pack_factor (== 2 for int4)
    should be rejected with a clear error and the MoE node left
    unchanged.
    """
    # Construct an MoE node with an odd fc1 second dimension.
    rng = np.random.RandomState(0)
    fc1 = rng.randn(2, 3, 8).astype(np.float32)  # E=2, N=3 (odd), K=8
    fc2 = rng.randn(2, 8, 4).astype(np.float32)
    inputs = [
        helper.make_tensor_value_info("x", TensorProto.FLOAT, [None, 8]),
        helper.make_tensor_value_info("r", TensorProto.FLOAT, [None, 2]),
    ]
    out = helper.make_tensor_value_info("y", TensorProto.FLOAT, [None, 8])
    moe = helper.make_node(
        "MoE",
        ["x", "r", "fc1_W", "", "fc2_W"],
        ["y"],
        name="m",
        domain="com.microsoft",
        k=1,
        activation_type="silu",
    )
    graph = helper.make_graph(
        [moe],
        "g",
        inputs,
        [out],
        initializer=[numpy_helper.from_array(fc1, "fc1_W"), numpy_helper.from_array(fc2, "fc2_W")],
    )
    model = helper.make_model(
        graph,
        opset_imports=[helper.make_opsetid("", 20), helper.make_opsetid("com.microsoft", 1)],
    )
    model.ir_version = 10
    p_in = tmp_path / "m.onnx"
    onnx.save(model, p_in)

    p = create_pass_from_dict(OnnxMoEQuantization, {"bits": 4}, disable_search=True)
    with patch(
        "olive.passes.onnx.moe_quantization._load_cuda_pack_fn",
        return_value=_fake_pack_weights_for_cuda_mixed_gemm,
    ):
        output_model = p.run(ONNXModelHandler(model_path=str(p_in)), str(tmp_path / "out"))
    g = output_model.load_model().graph
    assert [n.op_type for n in g.node] == ["MoE"], "Odd-N MoE node should be skipped, not crash."
