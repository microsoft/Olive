# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
from onnxscript import ir

from olive.model import ONNXModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.graph_surgeries import GraphSurgeries
from olive.passes.onnx.graph_surgery.base import Surgeon

_MS_DOMAIN = "com.microsoft"


def _attribute(name, value):
    if isinstance(value, float):
        return ir.AttrFloat32(name, value)
    if isinstance(value, int):
        return ir.AttrInt64(name, value)
    if isinstance(value, list):
        return ir.AttrInt64s(name, value)
    raise TypeError(f"Unsupported attribute value: {value!r}")


def _value(name, dtype=ir.DataType.FLOAT, shape=None):
    return ir.val(name, dtype=dtype, shape=shape)


def _initializer(name, array, dtype=None):
    array = np.asarray(array)
    if dtype is not None:
        array = array.astype(dtype.numpy())
    tensor = ir.tensor(array, dtype=dtype, name=name)
    return ir.val(name, dtype=tensor.dtype, shape=array.shape, const_value=tensor)


def _node(op_type, inputs, *, attributes=None, domain="", num_outputs=1, output_names=None):
    attrs = {name: _attribute(name, value) for name, value in (attributes or {}).items()}
    node = ir.Node(domain, op_type, inputs, attributes=attrs, num_outputs=num_outputs)
    if output_names:
        for output, name in zip(node.outputs, output_names):
            output.name = name
    return node


def _constant(name, value, dtype=None):
    array = np.asarray(value)
    if dtype is not None:
        array = array.astype(dtype.numpy())
    tensor = ir.tensor(array, dtype=dtype, name=f"{name}_value")
    node = ir.Node("", "Constant", [], attributes={"value": ir.AttrTensor("value", tensor)}, num_outputs=1)
    node.outputs[0].name = name
    node.outputs[0].dtype = tensor.dtype
    node.outputs[0].shape = ir.Shape(tensor.shape)
    node.outputs[0].const_value = tensor
    return node


def _model(inputs, outputs, nodes, initializers=()):
    graph = ir.Graph(
        inputs,
        outputs,
        nodes=nodes,
        initializers=initializers,
        name="attention_surgery_test",
        opset_imports={"": 24, _MS_DOMAIN: 1},
    )
    return ir.Model(graph, ir_version=10)


def _count_ops(model):
    return Counter(node.op_type for node in model.graph)


def _run_surgeries(tmp_path, model, *surgeons):
    input_path = tmp_path / "input.onnx"
    ir.save(model, input_path)
    olive_model = ONNXModelHandler(model_path=str(input_path))
    graph_surgeries = create_pass_from_dict(
        GraphSurgeries,
        {
            "surgeries": [{"surgeon": surgeon} for surgeon in surgeons],
            "remove_duplicate_initializers": False,
        },
        disable_search=True,
    )
    output_model = graph_surgeries.run(olive_model, str(tmp_path / "output"))
    return ir.load(output_model.model_path)


def _make_mask(nodes, *, prefix, sliding_window=None, bidirectional=False):
    query_index = _value(f"{prefix}_query_index", ir.DataType.INT64, [1, 1])
    key_index = _value(f"{prefix}_key_index", ir.DataType.INT64, [1, 1])
    causal = _node("GreaterOrEqual", [query_index, key_index], output_names=[f"{prefix}_causal"])
    nodes.append(causal)
    condition = causal.outputs[0]

    if sliding_window is not None:
        distance = _node("Sub", [query_index, key_index], output_names=[f"{prefix}_distance"])
        window = _constant(f"{prefix}_window", sliding_window, ir.DataType.INT64)
        within = _node("Less", [distance.outputs[0], window.outputs[0]], output_names=[f"{prefix}_within"])
        both = _node("And", [condition, within.outputs[0]], output_names=[f"{prefix}_visible"])
        nodes.extend([distance, window, within, both])
        condition = both.outputs[0]

    if bidirectional:
        same_block = _node("Equal", [query_index, key_index], output_names=[f"{prefix}_same_block"])
        overlay = _node("Or", [condition, same_block.outputs[0]], output_names=[f"{prefix}_overlay"])
        nodes.extend([same_block, overlay])
        condition = overlay.outputs[0]

    zero = _constant(f"{prefix}_zero", np.float32(0))
    negative = _constant(f"{prefix}_negative", np.float32(-10000))
    bias = _node("Where", [condition, zero.outputs[0], negative.outputs[0]], output_names=[f"{prefix}_bias"])
    nodes.extend([zero, negative, bias])
    return [query_index, key_index], bias.outputs[0]


def _make_attention_model(
    *,
    rotary=False,
    num_layers=1,
    sliding_window=None,
    bidirectional=False,
    with_cache=True,
    k_dim=32,
    v_dim=32,
):
    nodes = []
    inputs = [_value("attention_mask", ir.DataType.INT64, [1, "total_sequence"])]
    mask_inputs, attention_bias = _make_mask(
        nodes,
        prefix="mask",
        sliding_window=sliding_window,
        bidirectional=bidirectional,
    )
    inputs.extend(mask_inputs)

    cos_cache = _value("cos_cache", ir.DataType.FLOAT16, [128, 8])
    sin_cache = _value("sin_cache", ir.DataType.FLOAT16, [128, 8])
    position_ids = _value("position_ids", ir.DataType.INT64, [1, 2])
    if rotary:
        inputs.extend([cos_cache, sin_cache, position_ids])

    graph_outputs = []
    for layer in range(num_layers):
        q = _value(f"q_{layer}", ir.DataType.FLOAT16, [1, 2, 64])
        k = _value(f"k_{layer}", ir.DataType.FLOAT16, [1, 2, k_dim])
        v = _value(f"v_{layer}", ir.DataType.FLOAT16, [1, 2, v_dim])
        past_key = _value(f"past_key_{layer}", ir.DataType.FLOAT16, [1, 2, 4, k_dim])
        past_value = _value(f"past_value_{layer}", ir.DataType.FLOAT16, [1, 2, 4, v_dim])
        inputs.extend([q, k, v])
        if with_cache:
            inputs.extend([past_key, past_value])
        else:
            past_key = None
            past_value = None

        if rotary:
            cos = _node("Gather", [cos_cache, position_ids], output_names=[f"cos_{layer}"])
            sin = _node("Gather", [sin_cache, position_ids], output_names=[f"sin_{layer}"])
            q_rope = _node(
                "RotaryEmbedding",
                [q, cos.outputs[0], sin.outputs[0]],
                attributes={"interleaved": 1, "rotary_embedding_dim": 8},
                output_names=[f"q_rot_{layer}"],
            )
            k_rope = _node(
                "RotaryEmbedding",
                [k, cos.outputs[0], sin.outputs[0]],
                attributes={"interleaved": 1, "rotary_embedding_dim": 8},
                output_names=[f"k_rot_{layer}"],
            )
            nodes.extend([cos, sin, q_rope, k_rope])
            q_input = q_rope.outputs[0]
            k_input = k_rope.outputs[0]
        else:
            q_input = q
            k_input = k

        attention = _node(
            "Attention",
            [q_input, k_input, v, attention_bias, past_key, past_value],
            attributes={"scale": 0.125, "q_num_heads": 4, "kv_num_heads": 2, "softcap": 30.0},
            num_outputs=3,
            output_names=[f"output_{layer}", f"present_key_{layer}", f"present_value_{layer}"],
        )
        nodes.append(attention)
        for output, shape in zip(
            attention.outputs,
            ([1, 2, 64], [1, 2, 6, k_dim], [1, 2, 6, v_dim]),
        ):
            output.dtype = ir.DataType.FLOAT16
            output.shape = ir.Shape(shape)
        graph_outputs.extend(attention.outputs)

    return _model(inputs, graph_outputs, nodes)


def _make_gqa_projection_model(*, bias=False, computed_weight=False, dtype=ir.DataType.FLOAT16):
    hidden = _value("hidden", dtype, [1, 2, 8])
    past_key = _value("past_key", dtype, [1, 2, 4, 4])
    past_value = _value("past_value", dtype, [1, 2, 4, 4])
    seqlens = _value("seqlens_k", ir.DataType.INT32, [1])
    total_seq_len = _value("total_sequence_length", ir.DataType.INT32, [])
    inputs = [hidden, past_key, past_value, seqlens, total_seq_len]
    initializers = []
    nodes = []
    projections = []

    for name, output_size in (("q", 8), ("k", 4), ("v", 4)):
        array = np.arange(output_size * 8, dtype=np.float16).reshape(output_size, 8)
        weight = _initializer(f"{name}_weight", array, dtype)
        initializers.append(weight)
        weight_input = weight
        if computed_weight and name == "q":
            identity = _node("Identity", [weight], output_names=["computed_q_weight"])
            nodes.append(identity)
            weight_input = identity.outputs[0]
        transpose = _node("Transpose", [weight_input], attributes={"perm": [1, 0]}, output_names=[f"{name}_weight_t"])
        matmul = _node("MatMul", [hidden, transpose.outputs[0]], output_names=[f"{name}_matmul"])
        nodes.extend([transpose, matmul])
        projection = matmul.outputs[0]
        if bias:
            bias_value = _initializer(f"{name}_bias", np.arange(output_size, dtype=np.float16), dtype)
            initializers.append(bias_value)
            add = _node("Add", [projection, bias_value], output_names=[f"{name}_projection"])
            nodes.append(add)
            projection = add.outputs[0]
        projections.append(projection)

    gqa = _node(
        "GroupQueryAttention",
        [
            projections[0],
            projections[1],
            projections[2],
            past_key,
            past_value,
            seqlens,
            total_seq_len,
            None,
            None,
        ],
        domain=_MS_DOMAIN,
        attributes={"num_heads": 4, "kv_num_heads": 2, "scale": 0.5, "do_rotary": 0},
        num_outputs=3,
        output_names=["output", "present_key", "present_value"],
    )
    nodes.append(gqa)
    return _model(inputs, gqa.outputs, nodes, initializers)


def _make_packed_gqa_model(*, bias=False, legacy=False, invalid_size=False, dtype=ir.DataType.FLOAT16):
    hidden = _value("hidden", dtype, [1, 2, 8])
    past_key = _value("past_key", dtype, [1, 2, 4, 4])
    past_value = _value("past_value", dtype, [1, 2, 4, 4])
    seqlens = _value("seqlens_k", ir.DataType.INT32, [1])
    total_seq_len = _value("total_sequence_length", ir.DataType.INT32, [])
    inputs = [hidden, past_key, past_value, seqlens, total_seq_len]
    nodes = []
    initializers = []

    if legacy:
        output_size = 15 if invalid_size else 16
        packed_array = np.arange(output_size * 8, dtype=np.float16).reshape(output_size, 8)
        packed_weight = _initializer("packed_weight", packed_array, dtype)
        initializers.append(packed_weight)
    else:
        weights = []
        for name, output_size in (("q", 8), ("k", 4), ("v", 4)):
            weight = _initializer(
                f"{name}_weight",
                np.arange(output_size * 8, dtype=np.float16).reshape(output_size, 8),
                dtype,
            )
            initializers.append(weight)
            weights.append(weight)
        concat = _node("Concat", weights, attributes={"axis": 0}, output_names=["packed_weight"])
        nodes.append(concat)
        packed_weight = concat.outputs[0]

    transpose = _node("Transpose", [packed_weight], attributes={"perm": [1, 0]}, output_names=["packed_weight_t"])
    matmul = _node("MatMul", [hidden, transpose.outputs[0]], output_names=["packed_matmul"])
    nodes.extend([transpose, matmul])
    packed_projection = matmul.outputs[0]
    if bias:
        biases = []
        for name, output_size in (("q", 8), ("k", 4), ("v", 4)):
            bias_value = _initializer(f"{name}_bias", np.arange(output_size, dtype=np.float16), dtype)
            initializers.append(bias_value)
            biases.append(bias_value)
        bias_concat = _node("Concat", biases, attributes={"axis": 0}, output_names=["packed_bias"])
        add = _node("Add", [packed_projection, bias_concat.outputs[0]], output_names=["packed_projection"])
        nodes.extend([bias_concat, add])
        packed_projection = add.outputs[0]

    gqa = _node(
        "GroupQueryAttention",
        [packed_projection, None, None, past_key, past_value, seqlens, total_seq_len, None, None],
        domain=_MS_DOMAIN,
        attributes={"num_heads": 4, "kv_num_heads": 2, "scale": 0.5, "do_rotary": 0},
        num_outputs=3,
        output_names=["output", "present_key", "present_value"],
    )
    nodes.append(gqa)
    return _model(inputs, gqa.outputs, nodes, initializers)


def _make_fused_rope_gqa_model(*, do_rotary=1, include_caches=True, include_position_ids=True):
    dtype = ir.DataType.FLOAT16
    q = _value("q", dtype, [1, 2, 8])
    k = _value("k", dtype, [1, 2, 4])
    v = _value("v", dtype, [1, 2, 4])
    past_key = _value("past_key", dtype, [1, 2, 4, 4])
    past_value = _value("past_value", dtype, [1, 2, 4, 4])
    seqlens = _value("seqlens_k", ir.DataType.INT32, [1])
    total_seq_len = _value("total_sequence_length", ir.DataType.INT32, [])
    cos_cache = _value("cos_cache", dtype, [128, 2])
    sin_cache = _value("sin_cache", dtype, [128, 2])
    position_ids = _value("position_ids", ir.DataType.INT64, [1, 2])
    inputs = [q, k, v, past_key, past_value, seqlens, total_seq_len]
    if include_caches:
        inputs.extend([cos_cache, sin_cache])
    else:
        cos_cache = None
        sin_cache = None
    if include_position_ids:
        inputs.append(position_ids)

    gqa = _node(
        "GroupQueryAttention",
        [q, k, v, past_key, past_value, seqlens, total_seq_len, cos_cache, sin_cache],
        domain=_MS_DOMAIN,
        attributes={
            "num_heads": 4,
            "kv_num_heads": 2,
            "scale": 0.5,
            "do_rotary": do_rotary,
            "rotary_interleaved": 1,
            "softcap": 20.0,
        },
        num_outputs=3,
        output_names=["output", "present_key", "present_value"],
    )
    return _model(inputs, gqa.outputs, [gqa])


def _make_block_diagonal_attention_model(*, false_bias=-10000.0, shared_segments=True):
    dtype = ir.DataType.FLOAT16
    q = _value("q", dtype, [6, 8])
    k = _value("k", dtype, [6, 8])
    v = _value("v", dtype, [6, 8])
    positions = _value("positions", ir.DataType.INT64, [6, 1])
    cu_seqlens = _value("cu_seqlens", ir.DataType.INT64, [3])
    inputs = [q, k, v, positions, cu_seqlens]
    nodes = []

    axes_0 = _constant("axes_0", [0], ir.DataType.INT64)
    axes_1 = _constant("axes_1", [1], ir.DataType.INT64)
    cu_unsqueeze = _node("Unsqueeze", [cu_seqlens, axes_0.outputs[0]], output_names=["cu_unsqueeze"])
    greater_equal = _node("GreaterOrEqual", [positions, cu_unsqueeze.outputs[0]], output_names=["greater_equal"])
    cast = _node("Cast", [greater_equal.outputs[0]], attributes={"to": 7}, output_names=["cast"])
    reduce_sum = _node("ReduceSum", [cast.outputs[0], axes_1.outputs[0]], output_names=["reduce_sum"])
    one = _constant("one", np.int64(1), ir.DataType.INT64)
    segment_ids = _node("Sub", [reduce_sum.outputs[0], one.outputs[0]], output_names=["segment_ids"])
    row = _node("Unsqueeze", [segment_ids.outputs[0], axes_1.outputs[0]], output_names=["segment_row"])
    column_source = segment_ids.outputs[0] if shared_segments else reduce_sum.outputs[0]
    column = _node("Unsqueeze", [column_source, axes_0.outputs[0]], output_names=["segment_column"])
    equal = _node("Equal", [row.outputs[0], column.outputs[0]], output_names=["same_segment"])
    zero = _constant("zero", np.float16(0), dtype)
    negative = _constant("negative", np.float16(false_bias), dtype)
    bias = _node("Where", [equal.outputs[0], zero.outputs[0], negative.outputs[0]], output_names=["attention_bias"])
    q_4d = _node("Unsqueeze", [q, axes_0.outputs[0]], output_names=["q_4d"])
    k_4d = _node("Unsqueeze", [k, axes_0.outputs[0]], output_names=["k_4d"])
    v_4d = _node("Unsqueeze", [v, axes_0.outputs[0]], output_names=["v_4d"])
    axes_01 = _constant("axes_01", [0, 1], ir.DataType.INT64)
    bias_4d = _node("Unsqueeze", [bias.outputs[0], axes_01.outputs[0]], output_names=["bias_4d"])
    attention = _node(
        "Attention",
        [q_4d.outputs[0], k_4d.outputs[0], v_4d.outputs[0], bias_4d.outputs[0]],
        attributes={"scale": 0.5, "q_num_heads": 2},
        output_names=["attention_output"],
    )
    squeeze = _node("Squeeze", [attention.outputs[0], axes_0.outputs[0]], output_names=["output"])
    nodes.extend(
        [
            axes_0,
            axes_1,
            cu_unsqueeze,
            greater_equal,
            cast,
            reduce_sum,
            one,
            segment_ids,
            row,
            column,
            equal,
            zero,
            negative,
            bias,
            q_4d,
            k_4d,
            v_4d,
            axes_01,
            bias_4d,
            attention,
            squeeze,
        ]
    )
    squeeze.outputs[0].dtype = dtype
    squeeze.outputs[0].shape = ir.Shape([6, 8])
    return _model(inputs, [squeeze.outputs[0]], nodes)


def test_attention_surgeries_register_on_module_import():
    expected = {
        "attentiontogroupqueryattention",
        "separategroupqueryattentionrope",
        "unpackgroupqueryattentionqkv",
        "blockdiagonalattentiontopackedmha",
        "packqkvforgroupqueryattention",
    }
    assert expected <= Surgeon.registry.keys()


def test_attention_to_gqa_fuses_rotary_preserves_attributes_outputs_and_shared_inputs(tmp_path):
    model = _make_attention_model(rotary=True, num_layers=2, sliding_window=64, k_dim=32, v_dim=32)
    rewritten = _run_surgeries(tmp_path, model, "AttentionToGroupQueryAttention")

    counts = _count_ops(rewritten)
    assert counts["Attention"] == 0
    assert counts["RotaryEmbedding"] == 0
    assert counts["GroupQueryAttention"] == 2
    assert counts["ReduceSum"] == 1
    assert counts["Shape"] == 1

    gqa_nodes = [node for node in rewritten.graph if node.op_type == "GroupQueryAttention"]
    assert len(rewritten.graph.outputs) == 6
    assert [len(node.outputs) for node in gqa_nodes] == [3, 3]
    assert gqa_nodes[0].inputs[5].name == gqa_nodes[1].inputs[5].name
    assert gqa_nodes[0].inputs[6].name == gqa_nodes[1].inputs[6].name
    for node in gqa_nodes:
        assert node.inputs[0].shape[-1] == 64
        assert node.inputs[1].shape[-1] == 32
        assert node.attributes.get_int("do_rotary") == 1
        assert node.attributes.get_int("rotary_interleaved") == 1
        assert node.attributes.get_int("rotary_embedding_dim") == 8
        assert node.attributes.get_int("local_window_size") == 64
        assert node.attributes.get_float("softcap") == pytest.approx(30.0)
        assert node.inputs[7].name == "cos_cache"
        assert node.inputs[8].name == "sin_cache"


def test_attention_to_gqa_fallback_uses_external_rope_and_preserves_three_outputs(tmp_path):
    model = _make_attention_model(rotary=False, sliding_window=16)
    rewritten = _run_surgeries(tmp_path, model, "AttentionToGroupQueryAttention")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")

    assert gqa.attributes.get_int("do_rotary") == 0
    assert gqa.attributes.get_int("local_window_size") == 16
    assert len(gqa.inputs) == 7
    assert len(gqa.outputs) == 3
    assert len(rewritten.graph.outputs) == 3


@pytest.mark.parametrize(
    "model",
    [
        _make_attention_model(rotary=False, with_cache=False),
        _make_attention_model(rotary=False, k_dim=32, v_dim=16),
        _make_attention_model(rotary=False, bidirectional=True),
    ],
)
def test_attention_to_gqa_preserves_non_matches(tmp_path, model):
    rewritten = _run_surgeries(tmp_path, model, "AttentionToGroupQueryAttention")
    assert _count_ops(rewritten)["Attention"] == 1
    assert _count_ops(rewritten)["GroupQueryAttention"] == 0


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("dtype", [ir.DataType.FLOAT, ir.DataType.FLOAT16])
def test_pack_qkv_packs_bias_variants_preserves_dtype_order_tail_and_outputs(tmp_path, bias, dtype):
    model = _make_gqa_projection_model(bias=bias, dtype=dtype)
    rewritten = _run_surgeries(tmp_path, model, "PackQKVForGroupQueryAttention")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")

    assert gqa.inputs[1] is None
    assert gqa.inputs[2] is None
    assert [value.name if value else None for value in gqa.inputs[3:]] == [
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        None,
        None,
    ]
    assert len(gqa.outputs) == 3
    assert _count_ops(rewritten)["MatMul"] == 1

    packed_projection = gqa.inputs[0]
    if bias:
        add = packed_projection.producer()
        assert add.op_type == "Add"
        matmul = next(value.producer() for value in add.inputs if value.producer().op_type == "MatMul")
        bias_concat = next(value.producer() for value in add.inputs if value.producer().op_type == "Concat")
        assert [value.name for value in bias_concat.inputs] == ["q_bias", "k_bias", "v_bias"]
        assert bias_concat.outputs[0].dtype == dtype
    else:
        matmul = packed_projection.producer()

    transpose = matmul.inputs[1].producer()
    weight_concat = transpose.inputs[0].producer()
    assert [value.name for value in weight_concat.inputs] == ["q_weight", "k_weight", "v_weight"]
    assert weight_concat.outputs[0].dtype == dtype
    assert transpose.outputs[0].dtype == dtype


def test_pack_qkv_does_not_pack_computed_projection_weight(tmp_path):
    model = _make_gqa_projection_model(computed_weight=True)
    rewritten = _run_surgeries(tmp_path, model, "PackQKVForGroupQueryAttention")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")
    assert gqa.inputs[1] is not None
    assert gqa.inputs[2] is not None
    assert _count_ops(rewritten)["MatMul"] == 3


def test_separate_rope_adds_gathers_and_rotary_embeddings_and_preserves_attributes(tmp_path):
    model = _make_fused_rope_gqa_model()
    rewritten = _run_surgeries(tmp_path, model, "SeparateGroupQueryAttentionRoPE")
    counts = _count_ops(rewritten)
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")

    assert counts["Gather"] == 2
    assert counts["RotaryEmbedding"] == 2
    assert counts["GroupQueryAttention"] == 1
    assert gqa.attributes.get_int("do_rotary") == 0
    assert gqa.attributes.get_int("rotary_interleaved") == 1
    assert gqa.attributes.get_float("softcap") == pytest.approx(20.0)
    assert len(gqa.inputs) == 7
    assert len(gqa.outputs) == 3
    q_rope = gqa.inputs[0].producer()
    k_rope = gqa.inputs[1].producer()
    assert q_rope.attributes.get_int("num_heads") == 4
    assert k_rope.attributes.get_int("num_heads") == 2


@pytest.mark.parametrize(
    "model",
    [
        _make_fused_rope_gqa_model(do_rotary=0),
        _make_fused_rope_gqa_model(include_caches=False),
        _make_fused_rope_gqa_model(include_position_ids=False),
    ],
)
def test_separate_rope_preserves_non_matches(tmp_path, model):
    rewritten = _run_surgeries(tmp_path, model, "SeparateGroupQueryAttentionRoPE")
    assert _count_ops(rewritten)["RotaryEmbedding"] == 0
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")
    assert gqa.attributes.get_int("do_rotary") in (0, 1)
    assert len(gqa.inputs) == 9


@pytest.mark.parametrize("bias", [False, True])
def test_unpack_qkv_unpacks_concat_variants_and_preserves_optional_tail(tmp_path, bias):
    model = _make_packed_gqa_model(bias=bias)
    rewritten = _run_surgeries(tmp_path, model, "UnpackGroupQueryAttentionQKV")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")

    assert all(gqa.inputs[index] is not None for index in range(3))
    assert [value.name if value else None for value in gqa.inputs[3:]] == [
        "past_key",
        "past_value",
        "seqlens_k",
        "total_sequence_length",
        None,
        None,
    ]
    # The original packed projection was traced from the GQA input rather than
    # included in the match pattern, so it remains dead until unused-node cleanup.
    assert _count_ops(rewritten)["MatMul"] == 4
    expected_producer = "Add" if bias else "MatMul"
    assert [gqa.inputs[index].producer().op_type for index in range(3)] == [expected_producer] * 3
    assert len(gqa.outputs) == 3


def test_unpack_qkv_splits_legacy_initializer_with_unequal_q_and_kv_sizes(tmp_path):
    model = _make_packed_gqa_model(legacy=True)
    rewritten = _run_surgeries(tmp_path, model, "UnpackGroupQueryAttentionQKV")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")

    weights = [gqa.inputs[index].producer().inputs[1].producer().inputs[0] for index in range(3)]
    assert [list(weight.shape) for weight in weights] == [[8, 8], [4, 8], [4, 8]]
    assert [weight.dtype for weight in weights] == [ir.DataType.FLOAT16] * 3
    original = np.arange(16 * 8, dtype=np.float16).reshape(16, 8)
    np.testing.assert_array_equal(weights[0].const_value.numpy(), original[:8])
    np.testing.assert_array_equal(weights[1].const_value.numpy(), original[8:12])
    np.testing.assert_array_equal(weights[2].const_value.numpy(), original[12:])


@pytest.mark.parametrize(
    "model",
    [
        _make_gqa_projection_model(),
        _make_packed_gqa_model(legacy=True, invalid_size=True),
    ],
)
def test_unpack_qkv_preserves_non_matches(tmp_path, model):
    rewritten = _run_surgeries(tmp_path, model, "UnpackGroupQueryAttentionQKV")
    gqa = next(node for node in rewritten.graph if node.op_type == "GroupQueryAttention")
    if gqa.inputs[1] is None:
        assert _count_ops(rewritten)["MatMul"] == 1
    else:
        assert _count_ops(rewritten)["MatMul"] == 3


def test_block_diagonal_attention_rewrites_to_packed_mha_with_offsets(tmp_path):
    model = _make_block_diagonal_attention_model()
    rewritten = _run_surgeries(tmp_path, model, "BlockDiagonalAttentionToPackedMHA")
    counts = _count_ops(rewritten)
    packed = next(node for node in rewritten.graph if node.op_type == "PackedMultiHeadAttention")

    assert counts["Attention"] == 0
    assert counts["PackedMultiHeadAttention"] == 1
    assert packed.domain == _MS_DOMAIN
    assert [value.name if value else None for value in packed.inputs[:4]] == ["q", "k", "v", None]
    assert packed.inputs[4].producer().op_type == "Reshape"
    assert packed.inputs[5].producer().op_type == "Cast"
    assert packed.inputs[5].producer().inputs[0].name == "cu_seqlens"
    assert packed.attributes.get_float("scale") == pytest.approx(0.5)
    assert packed.attributes.get_int("num_heads") == 2
    assert len(rewritten.graph.outputs) == 1


@pytest.mark.parametrize(
    "model",
    [
        _make_block_diagonal_attention_model(false_bias=-1.0),
        _make_block_diagonal_attention_model(shared_segments=False),
    ],
)
def test_block_diagonal_attention_preserves_non_matches(tmp_path, model):
    rewritten = _run_surgeries(tmp_path, model, "BlockDiagonalAttentionToPackedMHA")
    assert _count_ops(rewritten)["Attention"] == 1
    assert _count_ops(rewritten)["PackedMultiHeadAttention"] == 0
