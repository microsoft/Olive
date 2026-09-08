# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Exporter-independent standard ONNX compatibility lowerings."""

from __future__ import annotations

import numpy as np
import onnx_ir as ir
from onnx_ir import tape
from onnxscript.rewriter import pattern

from olive.passes.onnx.graph_surgery import RewriteRuleSurgeon, Surgeon

_MASK_NEG = -3.0e38


class _BFloat16ClipRule(pattern.RewriteRuleClassBase):
    def check(self, context, x, **_):
        result = pattern.MatchResult()
        if x.dtype != ir.DataType.BFLOAT16:
            return result.fail("Clip input is not bfloat16")
        return result


class _ClipBothToMinMax(_BFloat16ClipRule):
    def pattern(self, op, x, lower, upper):
        return op.Clip(x, lower, upper)

    def rewrite(self, op, x, lower, upper):
        return op.Min(op.Max(x, lower), upper)


class _ClipMinToMax(_BFloat16ClipRule):
    def pattern(self, op, x, lower):
        return op.Clip(x, lower)

    def rewrite(self, op, x, lower):
        return op.Max(x, lower)


class ClipToMinMax(RewriteRuleSurgeon):
    """Lower bfloat16 Clip to Min and Max, which have ONNX Runtime kernels."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_ClipBothToMinMax().rule(), _ClipMinToMax().rule()])


class _Rank4RMSNormRule(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, weight):
        return op.RMSNormalization(x, weight, _allow_other_attributes=True, _outputs=["norm_out"])

    def check(self, context, x, norm_out, **_):
        result = pattern.MatchResult()
        if x.shape is None or len(x.shape) != 4:
            return result.fail("RMSNormalization input is not rank 4")
        heads, head_dim = x.shape[2], x.shape[3]
        if not isinstance(heads, int) or not isinstance(head_dim, int):
            return result.fail("RMSNormalization head dimensions are not static")

        node = norm_out.producer()
        if node.attributes.get_int("axis", -1) not in (-1, 3):
            return result.fail("RMSNormalization does not reduce only the last axis")
        if node.attributes.get_float("epsilon", None) is None:
            return result.fail("RMSNormalization has no epsilon attribute")
        return result

    def rewrite(self, op, x, weight, norm_out, **_):
        node = norm_out.producer()
        heads, head_dim = x.shape[2], x.shape[3]
        attributes = {name: attribute.value for name, attribute in node.attributes.items()}
        attributes["axis"] = -1

        rank3 = op.Reshape(x, op.Constant(value_ints=[0, -1, head_dim]))
        normalized = op.RMSNormalization(rank3, weight, **attributes)
        return op.Reshape(normalized, op.Constant(value_ints=[0, -1, heads, head_dim]))


class Rank4RMSNormToRank3(RewriteRuleSurgeon):
    """Reshape rank-4 RMSNormalization to an equivalent rank-3 normalization."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_Rank4RMSNormRule().rule()])


class _DecomposeOnnxRotaryEmbeddingRule(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, cos, sin):
        return op.RotaryEmbedding(x, cos, sin, _allow_other_attributes=True, _outputs=["rope_out"])

    def check(self, context, x, cos, sin, rope_out, **_):
        result = pattern.MatchResult()
        node = rope_out.producer()
        if node.domain not in ("", "ai.onnx"):
            return result.fail("RotaryEmbedding is not in the standard ONNX domain")
        if node.attributes.get_int("interleaved", 0) != 0:
            return result.fail("interleaved RotaryEmbedding is not supported")
        if node.attributes.get_int("rotary_embedding_dim", 0) != 0:
            return result.fail("partial RotaryEmbedding is not supported")
        if node.attributes.get_int("num_heads", 0) <= 0:
            return result.fail("num_heads must be positive")
        if x.shape is None or len(x.shape) != 3:
            return result.fail("RotaryEmbedding input is not rank 3")
        if cos.shape is None or len(cos.shape) != 3:
            return result.fail("RotaryEmbedding cosine input is not rank 3")
        if sin.shape is None or len(sin.shape) != 3:
            return result.fail("RotaryEmbedding sine input is not rank 3")
        return result

    def rewrite(self, op, x, cos, sin, rope_out, **_):
        num_heads = rope_out.producer().attributes.get_int("num_heads")

        # (B, S, N*H) -> (B, S, N, H), with H inferred for symbolic hidden sizes.
        x_rank4 = op.Reshape(x, op.Constant(value_ints=[0, 0, num_heads, -1]))
        half = op.Shape(cos, start=2, end=3)
        zero = op.Constant(value_ints=[0])
        last_axis = op.Constant(value_ints=[-1])
        int64_max = op.Constant(value_ints=[9223372036854775807])
        first_half = op.Slice(x_rank4, zero, half, last_axis)
        second_half = op.Slice(x_rank4, half, int64_max, last_axis)

        head_axis = op.Constant(value_ints=[2])
        cos = op.Unsqueeze(cos, head_axis)
        sin = op.Unsqueeze(sin, head_axis)
        rotated_first = op.Sub(op.Mul(first_half, cos), op.Mul(second_half, sin))
        rotated_second = op.Add(op.Mul(second_half, cos), op.Mul(first_half, sin))
        rotated = op.Concat(rotated_first, rotated_second, axis=-1)
        return op.Reshape(rotated, op.Constant(value_ints=[0, 0, -1]))


class DecomposeOnnxRotaryEmbedding(RewriteRuleSurgeon):
    """Decompose the standard ONNX RotaryEmbedding without changing the Microsoft ABI surgery."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_DecomposeOnnxRotaryEmbeddingRule().rule()])


class _TensorScatterToScatterNDRule(pattern.RewriteRuleClassBase):
    def pattern(self, op, cache, update, write_indices):
        return op.TensorScatter(
            cache,
            update,
            write_indices,
            _allow_other_attributes=True,
            _outputs=["scatter_out"],
        )

    def check(self, context, cache, scatter_out, **_):
        result = pattern.MatchResult()
        node = scatter_out.producer()
        if node.domain not in ("", "ai.onnx"):
            return result.fail("TensorScatter is not in the standard ONNX domain")
        if node.attributes.get_int("axis", 0) != 1:
            return result.fail("TensorScatter axis is not 1")
        if cache.shape is None or len(cache.shape) != 3:
            return result.fail("TensorScatter cache is not rank 3")
        if isinstance(cache.shape[0], int) and cache.shape[0] != 1:
            return result.fail("TensorScatter batch size is not 1")
        if not isinstance(cache.shape[1], int):
            return result.fail("TensorScatter cache length is not static")
        return result

    def rewrite(self, op, cache, update, write_indices, scatter_out, **_):
        max_length = int(cache.shape[1])
        zero = op.Constant(value_ints=[0])
        last_axis = op.Constant(value_ints=[-1])

        cache = op.Squeeze(cache, zero)
        update_rank2 = op.Squeeze(update, zero)
        full_range = op.Constant(value=ir.tensor(np.arange(max_length, dtype=np.int64)))
        sequence_length = op.Shape(update, start=1, end=2)
        offsets = op.Slice(full_range, zero, sequence_length, zero)
        start = op.Squeeze(write_indices, zero)
        positions = op.Add(offsets, start)
        indices = op.Unsqueeze(positions, last_axis)
        updated = op.ScatterND(cache, indices, update_rank2)
        return op.Unsqueeze(updated, zero)


class TensorScatterToScatterND(RewriteRuleSurgeon):
    """Lower batch-1 static-cache TensorScatter writes to ScatterND."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_TensorScatterToScatterNDRule().rule()])


def _is_decomposable_attention(node: ir.Node) -> bool:
    if node.op_type != "Attention" or node.domain not in ("", "ai.onnx"):
        return False
    query, key, value = node.inputs[:3]
    if any(
        input_value is None or input_value.shape is None or len(input_value.shape) != 3
        for input_value in (query, key, value)
    ):
        return False

    query_heads = node.attributes.get_int("q_num_heads", 0)
    kv_heads = node.attributes.get_int("kv_num_heads", 0)
    return (
        query_heads > 0
        and kv_heads > 0
        and query_heads % kv_heads == 0
        and node.attributes.get_int("qk_matmul_output_mode", 0) == 0
    )


def _constant_ints(builder: tape.Tape, values) -> ir.Value:
    return builder.op("Constant", [], {"value_ints": list(values)})


def _constant_floats(builder: tape.Tape, values) -> ir.Value:
    return builder.op("Constant", [], {"value_floats": [float(value) for value in values]})


def _attention_indices(builder: tape.Tape, length, static_length: int | None):
    if static_length is None:
        zero = builder.op("Constant", [], {"value_int": 0})
        one = builder.op("Constant", [], {"value_int": 1})
        return builder.op("Range", [zero, length, one])
    full_range = _constant_ints(builder, range(static_length))
    return builder.op(
        "Slice",
        [
            full_range,
            _constant_ints(builder, [0]),
            builder.op("Unsqueeze", [length, _constant_ints(builder, [0])]),
        ],
    )


def _static_kv_length(node: ir.Node) -> int | None:
    key = node.inputs[1]
    if key is None or key.shape is None or len(key.shape) != 3:
        return None
    key_length = key.shape[1]

    past_key = node.inputs[4] if len(node.inputs) > 4 else None
    if past_key is None:
        past_length = 0
    elif past_key.shape is None or len(past_key.shape) != 4 or not isinstance(past_key.shape[2], int):
        return None
    else:
        past_length = past_key.shape[2]
    return int(key_length) + int(past_length) if isinstance(key_length, int) else None


def _nonpadding_bias(builder: tape.Tape, nonpadding_length, key, scores, static_key_length):
    key_length = builder.op(
        "Squeeze",
        [
            builder.op(
                "Slice", [builder.op("Shape", [key]), _constant_ints(builder, [2]), _constant_ints(builder, [3])]
            ),
            _constant_ints(builder, [0]),
        ],
    )
    columns = _attention_indices(builder, key_length, static_key_length)
    columns = builder.op("Unsqueeze", [columns, _constant_ints(builder, [0, 1, 2])])
    limit = builder.op(
        "Unsqueeze",
        [builder.op("CastLike", [nonpadding_length, columns]), _constant_ints(builder, [1, 2, 3])],
    )
    allowed = builder.op("Less", [columns, limit])
    zero = builder.op("CastLike", [_constant_floats(builder, [0.0]), scores])
    masked = builder.op("CastLike", [_constant_floats(builder, [_MASK_NEG]), scores])
    return builder.op("Where", [allowed, zero, masked])


def _causal_bias(builder: tape.Tape, query, key, scores, static_key_length):
    query_length = builder.op(
        "Squeeze",
        [
            builder.op(
                "Slice", [builder.op("Shape", [query]), _constant_ints(builder, [2]), _constant_ints(builder, [3])]
            ),
            _constant_ints(builder, [0]),
        ],
    )
    key_length = builder.op(
        "Squeeze",
        [
            builder.op(
                "Slice", [builder.op("Shape", [key]), _constant_ints(builder, [2]), _constant_ints(builder, [3])]
            ),
            _constant_ints(builder, [0]),
        ],
    )
    rows = _attention_indices(builder, query_length, static_key_length)
    columns = _attention_indices(builder, key_length, static_key_length)
    offset = builder.op("Sub", [key_length, query_length])
    rows = builder.op("Add", [builder.op("Unsqueeze", [rows, _constant_ints(builder, [1])]), offset])
    columns = builder.op("Unsqueeze", [columns, _constant_ints(builder, [0])])
    allowed = builder.op("LessOrEqual", [columns, rows])
    zero = builder.op("CastLike", [_constant_floats(builder, [0.0]), scores])
    masked = builder.op("CastLike", [_constant_floats(builder, [_MASK_NEG]), scores])
    bias = builder.op("Where", [allowed, zero, masked])
    return builder.op("Unsqueeze", [bias, _constant_ints(builder, [0, 1])])


def _build_attention_replacement(node: ir.Node) -> tuple[list[ir.Node], list[ir.Value]]:
    builder = tape.Tape()
    inputs = node.inputs
    query, key, value = inputs[:3]
    attention_mask = inputs[3] if len(inputs) > 3 else None
    past_key = inputs[4] if len(inputs) > 4 else None
    past_value = inputs[5] if len(inputs) > 5 else None
    nonpadding_length = inputs[6] if len(inputs) > 6 else None

    query_heads = node.attributes.get_int("q_num_heads")
    kv_heads = node.attributes.get_int("kv_num_heads")
    group_size = query_heads // kv_heads
    scale = node.attributes.get_float("scale", None)
    softcap = node.attributes.get_float("softcap", 0.0) or 0.0
    is_causal = node.attributes.get_int("is_causal", 0)

    def split_heads(input_value, num_heads):
        rank4 = builder.op("Reshape", [input_value, _constant_ints(builder, [0, 0, num_heads, -1])])
        return builder.op("Transpose", [rank4], {"perm": [0, 2, 1, 3]})

    query = split_heads(query, query_heads)
    key = split_heads(key, kv_heads)
    value = split_heads(value, kv_heads)
    key = builder.op("Concat", [past_key, key], {"axis": 2}) if past_key is not None else key
    value = builder.op("Concat", [past_value, value], {"axis": 2}) if past_value is not None else value
    present_key, present_value = key, value

    def repeat_kv(input_value):
        if group_size == 1:
            return input_value
        rank5 = builder.op("Unsqueeze", [input_value, _constant_ints(builder, [2])])
        shape = builder.op("Shape", [input_value])
        batch = builder.op("Slice", [shape, _constant_ints(builder, [0]), _constant_ints(builder, [1])])
        heads = builder.op("Slice", [shape, _constant_ints(builder, [1]), _constant_ints(builder, [2])])
        sequence_and_head = builder.op("Slice", [shape, _constant_ints(builder, [2]), _constant_ints(builder, [4])])
        expand_shape = builder.op(
            "Concat",
            [batch, heads, _constant_ints(builder, [group_size]), sequence_and_head],
            {"axis": 0},
        )
        expanded = builder.op("Expand", [rank5, expand_shape])
        target_shape = builder.op(
            "Concat", [batch, _constant_ints(builder, [query_heads]), sequence_and_head], {"axis": 0}
        )
        return builder.op("Reshape", [expanded, target_shape])

    repeated_key = repeat_kv(key)
    repeated_value = repeat_kv(value)
    transposed_key = builder.op("Transpose", [repeated_key], {"perm": [0, 1, 3, 2]})
    scores = builder.op("MatMul", [query, transposed_key])
    if scale is not None:
        scale_value = builder.op("CastLike", [_constant_floats(builder, [scale]), scores])
        scores = builder.op("Mul", [scores, scale_value])
    else:
        head_dimension = builder.op(
            "Squeeze",
            [
                builder.op(
                    "Slice",
                    [builder.op("Shape", [query]), _constant_ints(builder, [3]), _constant_ints(builder, [4])],
                ),
                _constant_ints(builder, [0]),
            ],
        )
        head_dimension = builder.op("CastLike", [head_dimension, scores])
        scores = builder.op("Div", [scores, builder.op("Sqrt", [head_dimension])])

    if softcap:
        cap = builder.op("CastLike", [_constant_floats(builder, [softcap]), scores])
        scores = builder.op("Mul", [builder.op("Tanh", [builder.op("Div", [scores, cap])]), cap])
    if attention_mask is not None:
        scores = builder.op("Add", [scores, builder.op("CastLike", [attention_mask, scores])])

    static_key_length = _static_kv_length(node)
    if nonpadding_length is not None:
        scores = builder.op(
            "Add", [scores, _nonpadding_bias(builder, nonpadding_length, key, scores, static_key_length)]
        )
    if is_causal:
        scores = builder.op("Add", [scores, _causal_bias(builder, query, key, scores, static_key_length)])

    probabilities = builder.op("Softmax", [scores], {"axis": -1})
    output = builder.op("MatMul", [probabilities, repeated_value])
    output = builder.op("Transpose", [output], {"perm": [0, 2, 1, 3]})
    output = builder.op("Reshape", [output, _constant_ints(builder, [0, 0, -1])])
    return builder.nodes, [output, present_key, present_value]


class _DecomposeAttentionPass(ir.passes.InPlacePass):
    def call(self, model: ir.Model) -> ir.passes.PassResult:
        graph = model.graph
        targets = [node for node in graph if _is_decomposable_attention(node)]
        for node in targets:
            new_nodes, new_values = _build_attention_replacement(node)
            ir.convenience.replace_nodes_and_values(
                graph,
                insertion_point=node,
                old_nodes=[node],
                new_nodes=new_nodes,
                old_values=list(node.outputs),
                new_values=new_values[: len(node.outputs)],
            )
        return ir.passes.PassResult(model, modified=bool(targets))


class DecomposeAttention(Surgeon):
    """Lower standard ONNX Attention to scaled dot-product attention primitives."""

    def call_ir(self, model: ir.Model) -> ir.Model:
        return _DecomposeAttentionPass()(model).model


def _shape_tail_size(shape_tail: ir.Value) -> int | None:
    node = shape_tail.producer()
    if node is None or node.op_type != "Constant":
        return None
    value_ints = node.attributes.get("value_ints")
    if value_ints is None or len(value_ints.value) != 2 or value_ints.value[0] != 0:
        return None
    return int(value_ints.value[1])


def _static_empty_kv(op, kv_hidden: int, dtype: ir.DataType):
    return op.Constant(value=ir.tensor(np.zeros((1, 0, kv_hidden), dtype=dtype.numpy())))


class _StaticEmptyKVCastLikeRule(pattern.RewriteRuleClassBase):
    def pattern(self, op, query_states, shape_tail):
        batch_dimension = op.Shape(query_states, start=0, end=1)
        empty_shape = op.Concat(batch_dimension, shape_tail, axis=0)
        empty = op.ConstantOfShape(empty_shape)
        return op.CastLike(empty, query_states)

    def check(self, context, shape_tail, **_):
        result = pattern.MatchResult()
        if _shape_tail_size(shape_tail) is None:
            return result.fail("shape tail is not Constant(value_ints=[0, kv_hidden])")
        return result

    def rewrite(self, op, query_states, shape_tail, **_):
        return _static_empty_kv(
            op,
            _shape_tail_size(shape_tail),
            query_states.dtype or ir.DataType.FLOAT,
        )


class _StaticEmptyKVCastRule(pattern.RewriteRuleClassBase):
    def pattern(self, op, query_states, shape_tail):
        batch_dimension = op.Shape(query_states, start=0, end=1)
        empty_shape = op.Concat(batch_dimension, shape_tail, axis=0)
        empty = op.ConstantOfShape(empty_shape)
        return op.Cast(empty, _allow_other_attributes=True, _outputs=["empty_kv"])

    def check(self, context, shape_tail, empty_kv, **_):
        result = pattern.MatchResult()
        if _shape_tail_size(shape_tail) is None:
            return result.fail("shape tail is not Constant(value_ints=[0, kv_hidden])")
        if empty_kv.producer().attributes.get_int("to", None) is None:
            return result.fail("Cast has no target dtype")
        return result

    def rewrite(self, op, shape_tail, empty_kv, **_):
        dtype = ir.DataType(empty_kv.producer().attributes.get_int("to"))
        return _static_empty_kv(op, _shape_tail_size(shape_tail), dtype)


class StaticEmptyKV(RewriteRuleSurgeon):
    """Replace dynamic empty-KV construction with a static batch-1 Constant."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet(
            [
                _StaticEmptyKVCastLikeRule().rule(),
                _StaticEmptyKVCastRule().rule(),
            ]
        )


__all__ = [
    "ClipToMinMax",
    "DecomposeAttention",
    "DecomposeOnnxRotaryEmbedding",
    "Rank4RMSNormToRank3",
    "StaticEmptyKV",
    "TensorScatterToScatterND",
]
