# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Attention graph surgeries implemented with ONNXScript rewrite rules."""

from __future__ import annotations

import numpy as np
from onnxscript import ir
from onnxscript.rewriter import pattern
from onnxscript.rewriter._basics import MatchFailureError, MatchResult
from onnxscript.rewriter._rewrite_rule import RewriteRuleClassBase

from olive.constants import MSFT_DOMAIN
from olive.passes.onnx.graph_surgery.base import RewriteRuleSurgeon


def _initializer_dtype(value: ir.Value) -> ir.DataType | None:
    """Return an initializer's declared dtype, falling back to its tensor dtype."""
    declared = value.dtype
    const_dtype = value.const_value.dtype if value.const_value is not None else None
    if declared is not None and const_dtype is not None and declared != const_dtype:
        raise ValueError(
            f"Initializer {value.name!r} declares dtype {declared} but its const_value data is {const_dtype}."
        )
    return declared if declared is not None else const_dtype


def _propagate_dtype(source: ir.Value, *targets: ir.Value) -> None:
    dtype = _initializer_dtype(source)
    if dtype is not None:
        for target in targets:
            target.dtype = dtype


def _skip_view_ops(value: ir.Value | None) -> ir.Value | None:
    while value is not None:
        producer = value.producer()
        if producer is None or producer.op_type not in ("Cast", "Unsqueeze", "Identity"):
            return value
        value = producer.inputs[0]
    return value


def _constant_int(value: ir.Value | None) -> int | None:
    if value is None:
        return None
    tensor = value.const_value
    if tensor is None:
        producer = value.producer()
        if producer is None or producer.op_type != "Constant":
            return None
        attr = producer.attributes.get("value")
        tensor = getattr(attr, "value", None)
    if tensor is None:
        return None
    try:
        array = tensor.numpy()
    except Exception:  # pragma: no cover
        return None
    if array.size != 1:
        return None
    return int(array.reshape(-1)[0])


class _MaskShape:
    __slots__ = ("recognized", "window")

    def __init__(self, recognized: bool, window: int | None = None):
        self.recognized = recognized
        self.window = window


_MASK_WALK_LIMIT = 512


def _sliding_window_from_less(node: ir.Node) -> int | None:
    if len(node.inputs) < 2:
        return None
    distance = _skip_view_ops(node.inputs[0])
    distance_producer = distance.producer() if distance is not None else None
    if distance_producer is None or distance_producer.op_type != "Sub":
        return None
    window = _constant_int(node.inputs[1])
    if window is None or window <= 0:
        return None
    return window


def _local_window_from_attention_bias(attention_bias: ir.Value | None) -> _MaskShape:
    """Recover a sliding window and reject masks that GQA cannot represent."""
    window = None
    seen = set()
    stack = [attention_bias]
    visited = 0
    while stack:
        value = stack.pop()
        if value is None or id(value) in seen:
            continue
        seen.add(id(value))
        producer = value.producer()
        if producer is None:
            continue
        visited += 1
        if visited > _MASK_WALK_LIMIT:
            return _MaskShape(False)
        if producer.op_type == "Or":
            return _MaskShape(False)
        if producer.op_type == "Less":
            found = _sliding_window_from_less(producer)
            if found is not None:
                if window is not None and window != found:
                    return _MaskShape(False)
                window = found
                continue
        stack.extend(producer.inputs)
    return _MaskShape(True, window)


def _has_unequal_kv_head_dimensions(k, v, past_key, past_value) -> bool:
    def _static_last_dim(value):
        if value is None or value.shape is None or len(value.shape) == 0:
            return None
        dim = value.shape[-1]
        return dim if isinstance(dim, int) else None

    for key, value in ((k, v), (past_key, past_value)):
        key_dim = _static_last_dim(key)
        value_dim = _static_last_dim(value)
        if key_dim is not None and value_dim is not None and key_dim != value_dim:
            return True
    return False


class _RotaryAttentionToGQA(RewriteRuleClassBase):
    def __init__(self):
        super().__init__()
        self._seqlens_k = None
        self._total_seq_len = None
        self._cos_cache = None
        self._sin_cache = None

    def pattern(self, op, q_pre, k_pre, v, attention_bias, past_key, past_value, cos, sin):
        q_rot = op.RotaryEmbedding(q_pre, cos, sin, _allow_other_attributes=True)
        k_rot = op.RotaryEmbedding(k_pre, cos, sin, _allow_other_attributes=True)
        return op.Attention(
            q_rot,
            k_rot,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    def check(self, context, attn_out, k_pre, v, attention_bias, cos, sin, past_key, past_value, **_):
        result = MatchResult()
        if not _local_window_from_attention_bias(attention_bias).recognized:
            return result.fail("Attention bias cannot be represented by GroupQueryAttention")

        attn = attn_out.producer()
        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        cos_prod = cos.producer()
        sin_prod = sin.producer()
        if cos_prod is None or cos_prod.op_type != "Gather":
            return result.fail("cos must be Gather-produced")
        if sin_prod is None or sin_prod.op_type != "Gather":
            return result.fail("sin must be Gather-produced")

        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")
        if _has_unequal_kv_head_dimensions(k_pre, v, past_key, past_value):
            return result.fail("K and V head dimensions differ")
        return result

    def rewrite(
        self,
        op,
        q_pre,
        k_pre,
        v,
        attention_bias,
        past_key,
        past_value,
        cos,
        sin,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")
        softcap = attn.attributes.get_float("softcap", 0.0)

        q_rope_node = attn.inputs[0].producer()
        rotary_interleaved = q_rope_node.attributes.get_int("interleaved", 0)
        rotary_embedding_dim = q_rope_node.attributes.get_int("rotary_embedding_dim", 0)

        if self._cos_cache is None:
            self._cos_cache = cos.producer().inputs[0]
            self._sin_cache = sin.producer().inputs[0]

        if self._seqlens_k is None:
            attention_mask = next(gi for gi in attn.graph.inputs if gi.name == "attention_mask")
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(op.Sub(reduce_sum, one), to=6)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(op.Gather(mask_shape, idx_1), to=6)

        attrs = {
            "num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "scale": scale,
            "do_rotary": 1,
            "rotary_interleaved": rotary_interleaved,
        }
        if softcap:
            attrs["softcap"] = softcap
        if rotary_embedding_dim:
            attrs["rotary_embedding_dim"] = rotary_embedding_dim
        window = _local_window_from_attention_bias(attention_bias).window
        if window is not None:
            attrs["local_window_size"] = window

        outputs = op.GroupQueryAttention(
            q_pre,
            k_pre,
            v,
            past_key,
            past_value,
            self._seqlens_k,
            self._total_seq_len,
            self._cos_cache,
            self._sin_cache,
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class _AttentionToGQA(RewriteRuleClassBase):
    def __init__(self):
        super().__init__()
        self._seqlens_k = None
        self._total_seq_len = None

    def pattern(self, op, q, k, v, attention_bias, past_key, past_value):
        return op.Attention(
            q,
            k,
            v,
            attention_bias,
            past_key,
            past_value,
            _allow_other_attributes=True,
            _outputs=["attn_out", "present_key", "present_value"],
        )

    def check(self, context, attn_out, k, v, attention_bias, past_key, past_value, **_):
        result = MatchResult()
        if not _local_window_from_attention_bias(attention_bias).recognized:
            return result.fail("Attention bias cannot be represented by GroupQueryAttention")

        attn = attn_out.producer()
        if attn.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attn.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads on Attention")
        if attn.attributes.get_int("kv_num_heads", None) is None:
            return result.fail("Missing kv_num_heads on Attention")

        if past_key is None or past_value is None:
            return result.fail("No KV cache inputs")
        if past_key.producer() is not None:
            return result.fail("past_key is not a graph input")
        if past_value.producer() is not None:
            return result.fail("past_value is not a graph input")
        if not any(gi.name == "attention_mask" for gi in attn.graph.inputs):
            return result.fail("No attention_mask graph input")
        if _has_unequal_kv_head_dimensions(k, v, past_key, past_value):
            return result.fail("K and V head dimensions differ")
        return result

    def rewrite(
        self,
        op,
        q,
        k,
        v,
        attention_bias,
        past_key,
        past_value,
        attn_out,
        present_key,
        present_value,
        **_,
    ):
        attn = attn_out.producer()
        scale = attn.attributes.get_float("scale")
        q_num_heads = attn.attributes.get_int("q_num_heads")
        kv_num_heads = attn.attributes.get_int("kv_num_heads")
        softcap = attn.attributes.get_float("softcap", 0.0)

        if self._seqlens_k is None:
            attention_mask = next(gi for gi in attn.graph.inputs if gi.name == "attention_mask")
            axis = op.Constant(value_ints=[1])
            reduce_sum = op.ReduceSum(attention_mask, axis)
            one = op.Constant(value_ints=[1])
            self._seqlens_k = op.Cast(op.Sub(reduce_sum, one), to=6)
            mask_shape = op.Shape(attention_mask)
            idx_1 = op.Constant(value_int=1)
            self._total_seq_len = op.Cast(op.Gather(mask_shape, idx_1), to=6)

        attrs = {
            "num_heads": q_num_heads,
            "kv_num_heads": kv_num_heads,
            "scale": scale,
            "do_rotary": 0,
        }
        if softcap:
            attrs["softcap"] = softcap
        window = _local_window_from_attention_bias(attention_bias).window
        if window is not None:
            attrs["local_window_size"] = window

        outputs = op.GroupQueryAttention(
            q,
            k,
            v,
            past_key,
            past_value,
            self._seqlens_k,
            self._total_seq_len,
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class AttentionToGroupQueryAttention(RewriteRuleSurgeon):
    """Fuse decoder Attention, and standard RoPE when possible, into GQA."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_RotaryAttentionToGQA.rule(), _AttentionToGQA.rule()])


class _PackQKVForGQA(RewriteRuleClassBase):
    def pattern(self, op, hidden, q_w, k_w, v_w):
        q = op.MatMul(hidden, op.Transpose(q_w, perm=[1, 0]))
        k = op.MatMul(hidden, op.Transpose(k_w, perm=[1, 0]))
        v = op.MatMul(hidden, op.Transpose(v_w, perm=[1, 0]))
        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain=MSFT_DOMAIN,
            _allow_other_attributes=True,
            _allow_other_inputs=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    def check(self, context, q_w, k_w, v_w, **_):
        for name, weight in (("q_w", q_w), ("k_w", k_w), ("v_w", v_w)):
            if weight.producer() is not None:
                raise MatchFailureError(f"{name} {weight.name!r} is not a graph parameter")
        return True

    def rewrite(self, op, hidden, q_w, k_w, v_w, gqa_out, present_key, present_value, **_):
        packed_w = op.Concat(q_w, k_w, v_w, axis=0)
        packed_wt = op.Transpose(packed_w, perm=[1, 0])
        _propagate_dtype(q_w, packed_w, packed_wt)
        packed_qkv = op.MatMul(hidden, packed_wt)

        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        outputs = op.GroupQueryAttention(
            packed_qkv,
            None,
            None,
            *gqa_node.inputs[3:],
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class _PackQKVWithBiasForGQA(RewriteRuleClassBase):
    def pattern(self, op, hidden, q_w, bias_q, k_w, bias_k, v_w, bias_v):
        q = op.Add(op.MatMul(hidden, op.Transpose(q_w, perm=[1, 0])), bias_q)
        k = op.Add(op.MatMul(hidden, op.Transpose(k_w, perm=[1, 0])), bias_k)
        v = op.Add(op.MatMul(hidden, op.Transpose(v_w, perm=[1, 0])), bias_v)
        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain=MSFT_DOMAIN,
            _allow_other_attributes=True,
            _allow_other_inputs=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    def check(self, context, q_w, bias_q, k_w, bias_k, v_w, bias_v, **_):
        for name, weight in (("q_w", q_w), ("k_w", k_w), ("v_w", v_w)):
            if weight.producer() is not None:
                raise MatchFailureError(f"{name} {weight.name!r} is not a graph parameter")
        for bias in (bias_q, bias_k, bias_v):
            if bias is None or bias.producer() is not None:
                raise MatchFailureError(f"bias {bias!r} is not a graph parameter")
        return True

    def rewrite(
        self,
        op,
        hidden,
        q_w,
        bias_q,
        k_w,
        bias_k,
        v_w,
        bias_v,
        gqa_out,
        present_key,
        present_value,
        **_,
    ):
        packed_w = op.Concat(q_w, k_w, v_w, axis=0)
        packed_wt = op.Transpose(packed_w, perm=[1, 0])
        _propagate_dtype(q_w, packed_w, packed_wt)
        packed_mm = op.MatMul(hidden, packed_wt)

        packed_bias = op.Concat(bias_q, bias_k, bias_v, axis=0)
        _propagate_dtype(bias_q, packed_bias)
        packed_qkv = op.Add(packed_mm, packed_bias)

        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        outputs = op.GroupQueryAttention(
            packed_qkv,
            None,
            None,
            *gqa_node.inputs[3:],
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class PackQKVForGroupQueryAttention(RewriteRuleSurgeon):
    """Pack separate Q/K/V projections, with or without bias, for GQA."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_PackQKVForGQA.rule(), _PackQKVWithBiasForGQA.rule()])


class _SeparateGroupQueryAttentionRoPE(RewriteRuleClassBase):
    def pattern(self, op, q, k, v):
        return op.GroupQueryAttention(
            q,
            k,
            v,
            _domain=MSFT_DOMAIN,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    def check(self, context, q, k, v, gqa_out, **_):
        result = MatchResult()
        gqa_node = gqa_out.producer()
        if gqa_node is None:
            return result.fail("No GQA producer")
        if gqa_node.attributes.get_int("do_rotary", 0) != 1:
            return result.fail("do_rotary is not 1")
        if len(gqa_node.inputs) < 9:
            return result.fail("GQA has fewer than 9 inputs")
        if gqa_node.inputs[7] is None or gqa_node.inputs[8] is None:
            return result.fail("cos_cache or sin_cache is absent")
        if not any(graph_input.name == "position_ids" for graph_input in gqa_node.graph.inputs):
            return result.fail("No position_ids graph input")
        return result

    def rewrite(self, op, q, k, v, gqa_out, present_key, present_value, **_):
        gqa_node = gqa_out.producer()
        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        attrs["do_rotary"] = 0
        num_heads = attrs.get("num_heads", 1)
        kv_num_heads = attrs.get("kv_num_heads", 1)

        cos_cache = gqa_node.inputs[7]
        sin_cache = gqa_node.inputs[8]
        position_ids = next(graph_input for graph_input in gqa_node.graph.inputs if graph_input.name == "position_ids")
        gathered_cos = op.Gather(cos_cache, position_ids)
        gathered_sin = op.Gather(sin_cache, position_ids)
        q_rot = op.RotaryEmbedding(q, gathered_cos, gathered_sin, num_heads=num_heads)
        k_rot = op.RotaryEmbedding(k, gathered_cos, gathered_sin, num_heads=kv_num_heads)

        outputs = op.GroupQueryAttention(
            q_rot,
            k_rot,
            v,
            *gqa_node.inputs[3:7],
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class SeparateGroupQueryAttentionRoPE(RewriteRuleSurgeon):
    """Move fused rotary embedding out of GroupQueryAttention."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_SeparateGroupQueryAttentionRoPE.rule()])


class _UnpackGroupQueryAttentionQKV(RewriteRuleClassBase):
    def __init__(self):
        super().__init__()
        self._counter = 0
        self._split_weights = None
        self._bias_concat_node = None

    def pattern(self, op, packed_qkv):
        return op.GroupQueryAttention(
            packed_qkv,
            _domain=MSFT_DOMAIN,
            _allow_other_inputs=True,
            _allow_other_attributes=True,
            _outputs=["gqa_out", "present_key", "present_value"],
        )

    def check(self, context, packed_qkv, gqa_out, **_):
        result = MatchResult()
        gqa_node = gqa_out.producer()
        if gqa_node is None:
            return result.fail("No GQA producer")
        if len(gqa_node.inputs) < 3:
            return result.fail("GQA has fewer than 3 inputs")
        if gqa_node.inputs[1] is not None or gqa_node.inputs[2] is not None:
            return result.fail("GQA is not in packed mode")

        self._split_weights = None
        self._bias_concat_node = None
        qkv_producer = packed_qkv.producer()
        if qkv_producer is not None and qkv_producer.op_type == "Add":
            if len(qkv_producer.inputs) < 2 or None in qkv_producer.inputs:
                return result.fail("Add has missing inputs")
            left, right = qkv_producer.inputs[0], qkv_producer.inputs[1]
            left_prod = left.producer()
            right_prod = right.producer()
            if left_prod is not None and left_prod.op_type == "MatMul":
                matmul, bias_value = left_prod, right
            elif right_prod is not None and right_prod.op_type == "MatMul":
                matmul, bias_value = right_prod, left
            else:
                return result.fail("Add inputs do not include a MatMul")

            bias_concat = bias_value.producer() if bias_value else None
            if bias_concat is None or bias_concat.op_type != "Concat":
                return result.fail("Bias is not produced by Concat")
            if len(bias_concat.inputs) != 3:
                return result.fail("Bias Concat does not have exactly 3 inputs")
            for bias_input in bias_concat.inputs:
                if bias_input is None or bias_input.producer() is not None:
                    return result.fail("Bias Concat input is not a graph parameter")
            self._bias_concat_node = bias_concat
        else:
            matmul = qkv_producer
            if matmul is None or matmul.op_type != "MatMul":
                return result.fail("packed_qkv is not produced by MatMul")

        if len(matmul.inputs) < 2 or matmul.inputs[1] is None:
            return result.fail("MatMul has no weight input")
        transpose = matmul.inputs[1].producer()
        if transpose is None or transpose.op_type != "Transpose":
            return result.fail("MatMul weight is not produced by Transpose")
        packed_weight = transpose.inputs[0]
        if packed_weight is None:
            return result.fail("Transpose input is absent")

        concat_node = packed_weight.producer()
        if concat_node is not None and concat_node.op_type == "Concat":
            if len(concat_node.inputs) != 3:
                return result.fail("Concat does not have exactly 3 inputs")
            for weight in concat_node.inputs:
                if weight is None or weight.producer() is not None:
                    return result.fail("Concat input is not a graph parameter")
            return result

        weight_tensor = ir.convenience.get_const_tensor(packed_weight)
        if weight_tensor is None:
            return result.fail("Packed weight is neither Concat nor a constant")
        num_heads = gqa_node.attributes.get_int("num_heads", None)
        kv_num_heads = gqa_node.attributes.get_int("kv_num_heads", None)
        if num_heads is None or kv_num_heads is None:
            return result.fail("Missing num_heads or kv_num_heads")

        weight_array = weight_tensor.numpy()
        total_out = weight_array.shape[0]
        total_heads = num_heads + 2 * kv_num_heads
        if total_out % total_heads != 0:
            return result.fail(f"Cannot determine head_dim from {total_out} outputs and {total_heads} heads")

        head_dim = total_out // total_heads
        q_size = num_heads * head_dim
        k_size = kv_num_heads * head_dim
        self._split_weights = (
            weight_array[:q_size, :],
            weight_array[q_size : q_size + k_size, :],
            weight_array[q_size + k_size :, :],
        )
        return result

    def rewrite(self, op, packed_qkv, gqa_out, present_key, present_value, **_):
        gqa_node = gqa_out.producer()
        qkv_producer = packed_qkv.producer()
        if qkv_producer is not None and qkv_producer.op_type == "Add":
            left_prod = qkv_producer.inputs[0].producer()
            matmul = (
                left_prod
                if left_prod is not None and left_prod.op_type == "MatMul"
                else qkv_producer.inputs[1].producer()
            )
        else:
            matmul = qkv_producer
        hidden_states = matmul.inputs[0]
        packed_weight = matmul.inputs[1].producer().inputs[0]
        concat_node = packed_weight.producer()

        self._counter += 1
        suffix = self._counter
        if concat_node is not None and concat_node.op_type == "Concat":
            w_q, w_k, w_v = concat_node.inputs
            q_mm = op.MatMul(hidden_states, op.Transpose(w_q, perm=[1, 0]))
            k_mm = op.MatMul(hidden_states, op.Transpose(w_k, perm=[1, 0]))
            v_mm = op.MatMul(hidden_states, op.Transpose(w_v, perm=[1, 0]))
        else:
            w_q_array, w_k_array, w_v_array = self._split_weights
            self._split_weights = None

            def _projection(weight: np.ndarray, name: str) -> ir.Value:
                initializer = op.initializer(ir.tensor(weight, name=name), name=name)
                return op.MatMul(hidden_states, op.Transpose(initializer, perm=[1, 0]))

            q_mm = _projection(w_q_array, f"unpack_q_weight_{suffix}")
            k_mm = _projection(w_k_array, f"unpack_k_weight_{suffix}")
            v_mm = _projection(w_v_array, f"unpack_v_weight_{suffix}")

        bias_concat = self._bias_concat_node
        self._bias_concat_node = None
        if bias_concat is not None:
            bias_q, bias_k, bias_v = bias_concat.inputs
            q = op.Add(q_mm, bias_q)
            k = op.Add(k_mm, bias_k)
            v = op.Add(v_mm, bias_v)
        else:
            q, k, v = q_mm, k_mm, v_mm

        attrs = {key: gqa_node.attributes[key].value for key in gqa_node.attributes}
        outputs = op.GroupQueryAttention(
            q,
            k,
            v,
            *gqa_node.inputs[3:],
            _domain=MSFT_DOMAIN,
            _outputs=3,
            **attrs,
        )
        return outputs[0], outputs[1], outputs[2]


class UnpackGroupQueryAttentionQKV(RewriteRuleSurgeon):
    """Split packed GQA QKV projections into separate projections."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_UnpackGroupQueryAttentionQKV.rule()])


def _build_packed_token_offset(op, cu_seqlens):
    axes_0 = op.Constant(value_ints=[0])
    axes_1 = op.Constant(value_ints=[1])
    neg_one = op.Constant(value_ints=[-1])
    start_1 = op.Constant(value_ints=[1])
    int_max = op.Constant(value_ints=[np.iinfo(np.int64).max])

    cu_seqlens_i32 = op.Cast(cu_seqlens, to=ir.DataType.INT32)
    batch_size = op.Cast(op.Sub(op.Size(cu_seqlens), op.Constant(value_int=1)), to=ir.DataType.INT32)
    starts = op.Slice(cu_seqlens_i32, axes_0, neg_one, axes_0)
    ends = op.Slice(cu_seqlens_i32, start_1, int_max, axes_0)
    lengths = op.Sub(ends, starts)
    max_len = op.Squeeze(op.ReduceMax(lengths), axes_0)

    zero_i32 = op.Cast(op.Constant(value_int=0), to=ir.DataType.INT32)
    one_i32 = op.Cast(op.Constant(value_int=1), to=ir.DataType.INT32)
    rows = op.Range(zero_i32, batch_size, one_i32)
    cols = op.Range(zero_i32, max_len, one_i32)
    pos_matrix = op.Add(op.Mul(op.Unsqueeze(rows, axes_1), max_len), op.Unsqueeze(cols, axes_0))
    pos_matrix_shape = op.Shape(pos_matrix)

    valid_mask = op.Less(op.Unsqueeze(cols, axes_0), op.Unsqueeze(lengths, axes_1))
    valid_mask_1d = op.Reshape(valid_mask, neg_one)
    pos_1d = op.Reshape(pos_matrix, neg_one)
    valid_indices = op.Compress(pos_1d, valid_mask_1d)
    padding_indices = op.Compress(pos_1d, op.Not(valid_mask_1d))
    return op.Reshape(op.Concat(valid_indices, padding_indices, axis=0), pos_matrix_shape)


class _BlockDiagonalAttentionToPackedMHA(RewriteRuleClassBase):
    def pattern(self, op, q, k, v, attn_bias_2d):
        q_4d = op.Unsqueeze(q, [0])
        k_4d = op.Unsqueeze(k, [0])
        v_4d = op.Unsqueeze(v, [0])
        attn_bias_4d = op.Unsqueeze(attn_bias_2d, [0, 1])
        attn_out = op.Attention(
            q_4d,
            k_4d,
            v_4d,
            attn_bias_4d,
            _allow_other_attributes=True,
            _outputs=["attn_out"],
        )
        return op.Squeeze(attn_out, [0])

    def check(self, context, attn_bias_2d, attn_out, **_):
        result = MatchResult()
        where = attn_bias_2d.producer()
        if where is None or where.op_type != "Where":
            return result.fail("Expected Where producing attention bias")
        equal = where.inputs[0].producer()
        if equal is None or equal.op_type != "Equal":
            return result.fail("Expected Equal in mask condition")
        unsqueeze_row = equal.inputs[0].producer()
        unsqueeze_col = equal.inputs[1].producer()
        if not (
            unsqueeze_row
            and unsqueeze_col
            and unsqueeze_row.op_type == "Unsqueeze"
            and unsqueeze_col.op_type == "Unsqueeze"
        ):
            return result.fail("Expected Unsqueeze ops feeding Equal")
        if unsqueeze_row.inputs[0] is not unsqueeze_col.inputs[0]:
            return result.fail("Unsqueezes do not share segment_ids")

        segment_ids = unsqueeze_row.inputs[0]
        sub = segment_ids.producer()
        if sub is None or sub.op_type != "Sub":
            return result.fail("Expected Sub in segment_ids chain")
        reduce_sum = sub.inputs[0].producer()
        if reduce_sum is None or reduce_sum.op_type != "ReduceSum":
            return result.fail("Expected ReduceSum in segment_ids chain")
        cast = reduce_sum.inputs[0].producer()
        if cast is None or cast.op_type != "Cast":
            return result.fail("Expected Cast in segment_ids chain")
        greater_equal = cast.inputs[0].producer()
        if greater_equal is None or greater_equal.op_type != "GreaterOrEqual":
            return result.fail("Expected GreaterOrEqual in segment_ids chain")
        cu_unsqueeze = greater_equal.inputs[1].producer()
        if cu_unsqueeze is None or cu_unsqueeze.op_type != "Unsqueeze":
            return result.fail("Expected Unsqueeze for cu_seqlens")

        true_value = ir.convenience.get_const_tensor(where.inputs[1])
        false_value = ir.convenience.get_const_tensor(where.inputs[2])
        if true_value is None or not np.isclose(float(true_value.numpy()), 0.0, atol=1e-3):
            return result.fail("Where true branch must be approximately zero")
        if false_value is None or float(false_value.numpy()) > -100:
            return result.fail("Where false branch must be large negative")

        attention = attn_out.producer()
        if attention.attributes.get_float("scale", None) is None:
            return result.fail("Missing scale attribute on Attention")
        if attention.attributes.get_int("q_num_heads", None) is None:
            return result.fail("Missing q_num_heads attribute on Attention")
        return result

    @staticmethod
    def _trace_cu_seqlens(attn_bias_2d):
        where = attn_bias_2d.producer()
        equal = where.inputs[0].producer()
        segment_ids = equal.inputs[0].producer().inputs[0]
        sub = segment_ids.producer()
        reduce_sum = sub.inputs[0].producer()
        cast = reduce_sum.inputs[0].producer()
        greater_equal = cast.inputs[0].producer()
        return greater_equal.inputs[1].producer().inputs[0]

    def rewrite(self, op, q, k, v, attn_bias_2d, attn_out, **_):
        attention = attn_out.producer()
        scale = attention.attributes.get_float("scale")
        num_heads = attention.attributes.get_int("q_num_heads")
        cu_seqlens = self._trace_cu_seqlens(attn_bias_2d)
        cu_seqlens_i32 = op.Cast(cu_seqlens, to=6)
        token_offset = _build_packed_token_offset(op, cu_seqlens)
        return op.op(
            "PackedMultiHeadAttention",
            inputs=[q, k, v, None, token_offset, cu_seqlens_i32],
            domain=MSFT_DOMAIN,
            attributes={"scale": scale, "num_heads": num_heads},
        )


class BlockDiagonalAttentionToPackedMHA(RewriteRuleSurgeon):
    """Replace block-diagonal Attention with PackedMultiHeadAttention."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_BlockDiagonalAttentionToPackedMHA.rule()])
