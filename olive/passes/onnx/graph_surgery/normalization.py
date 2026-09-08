# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math

from onnxscript import ir
from onnxscript.rewriter import pattern

from olive.constants import MSFT_DOMAIN
from olive.passes.onnx.graph_surgery.base import RewriteRuleSurgeon

_SKIP_NORM_DTYPES = {ir.DataType.FLOAT, ir.DataType.FLOAT16, ir.DataType.BFLOAT16}


def _check_scalar_constant(value, expected: float, name: str) -> str | None:
    if value.const_value is None:
        return f"{name} is not a constant"
    array = value.const_value.numpy()
    if array.size != 1:
        return f"{name} must contain exactly one element"
    actual = float(array.flat[0])
    if not math.isclose(actual, expected, rel_tol=1e-4):
        return f"{name} is {actual}, expected {expected}"
    return None


def _check_last_axis(value, name: str) -> str | None:
    if value.const_value is None:
        return f"ReduceMean {name} axes is not a constant"
    axes = list(value.const_value.numpy().flat)
    if axes != [-1]:
        return f"ReduceMean {name} axes={axes}, expected [-1]"
    return None


def _check_layer_normalization_constants(exponent, epsilon, first_axes, second_axes):
    result = pattern.MatchResult()
    if error := _check_scalar_constant(exponent, 2.0, "Pow exponent"):
        return result.fail(error)

    if epsilon.const_value is None:
        return result.fail("Epsilon is not a constant")
    epsilon_array = epsilon.const_value.numpy()
    if epsilon_array.size != 1:
        return result.fail("Epsilon must contain exactly one element")
    epsilon_value = float(epsilon_array.flat[0])
    if epsilon_value <= 0 or epsilon_value > 1.0:
        return result.fail(f"Epsilon {epsilon_value} is outside the expected range (0, 1]")

    for name, axes in (("first", first_axes), ("second", second_axes)):
        if error := _check_last_axis(axes, name):
            return result.fail(error)
    return result


class _RemoveReplacedAdd:
    def setup(self):
        self._replaced_adds = []

    def _record_replaced_add(self, add):
        self._replaced_adds.append(add)

    def cleanup(self):
        for add in self._replaced_adds:
            if (
                add.graph is not None
                and all(output not in add.graph.outputs for output in add.outputs)
                and all(not list(output.uses()) for output in add.outputs)
            ):
                add.graph.remove(add, safe=True)


class _LayerNormalization(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, first_axes, exponent, second_axes, epsilon, weight, bias):
        mean = op.ReduceMean(x, first_axes, _allow_other_attributes=True)
        difference = op.Sub(x, mean)
        squared = op.Pow(difference, exponent)
        variance = op.ReduceMean(squared, second_axes, _allow_other_attributes=True)
        variance_epsilon = op.Add(variance, epsilon)
        standard_deviation = op.Sqrt(variance_epsilon)
        normalized = op.Div(difference, standard_deviation)
        scaled = op.Mul(normalized, weight)
        return op.Add(scaled, bias)

    def check(self, context, exponent, epsilon, first_axes, second_axes, **_):
        return _check_layer_normalization_constants(exponent, epsilon, first_axes, second_axes)

    def rewrite(self, op, x, weight, bias, epsilon, **_):
        epsilon_value = float(epsilon.const_value.numpy().flat[0])
        return op.LayerNormalization(x, weight, bias, axis=-1, epsilon=epsilon_value)


class _LayerNormalizationNoBias(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, first_axes, exponent, second_axes, epsilon, weight):
        mean = op.ReduceMean(x, first_axes, _allow_other_attributes=True)
        difference = op.Sub(x, mean)
        squared = op.Pow(difference, exponent)
        variance = op.ReduceMean(squared, second_axes, _allow_other_attributes=True)
        variance_epsilon = op.Add(variance, epsilon)
        standard_deviation = op.Sqrt(variance_epsilon)
        normalized = op.Div(difference, standard_deviation)
        return op.Mul(normalized, weight, _outputs=["norm_output"])

    def check(self, context, exponent, epsilon, first_axes, second_axes, norm_output, **_):
        result = _check_layer_normalization_constants(exponent, epsilon, first_axes, second_axes)
        if not result:
            return result
        if list(norm_output.uses()):
            return result.fail("Bias-free LayerNormalization output has another node consumer")
        return result

    def rewrite(self, op, x, weight, epsilon, **_):
        epsilon_value = float(epsilon.const_value.numpy().flat[0])
        return op.LayerNormalization(x, weight, axis=-1, epsilon=epsilon_value)


class FuseLayerNormalization(RewriteRuleSurgeon):
    """Fuse the ReduceMean-based LayerNormalization decomposition."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_LayerNormalization.rule(), _LayerNormalizationNoBias.rule()])


def _same_dimensions(first_shape, second_shape) -> bool:
    return len(first_shape) == len(second_shape) and all(
        first is not None and second is not None and first == second for first, second in zip(first_shape, second_shape)
    )


def _skip_normalization_inputs(add):
    first, second = add.inputs
    first_shape = first.shape
    second_shape = second.shape
    if first_shape is None or second_shape is None:
        return None, None, "Add input shapes must be known"

    first_rank = len(first_shape)
    second_rank = len(second_shape)
    if first_rank not in (2, 3) or second_rank not in (2, 3):
        return None, None, "Add input ranks must both be 2 or 3"

    if first_rank == 2 and second_rank == 2:
        if not _same_dimensions(first_shape, second_shape):
            return None, None, "Rank-2 Add inputs must have identical shapes"
        return first, second, None

    if first_rank != second_rank:
        data, skip = (first, second) if first_rank == 3 else (second, first)
        if not _same_dimensions(data.shape[1:], skip.shape):
            return None, None, "Rank-2 skip shape must match rank-3 data sequence and hidden dimensions"
        return data, skip, None

    if not _same_dimensions(first_shape[1:], second_shape[1:]):
        return None, None, "Rank-3 Add inputs must have matching sequence and hidden dimensions"
    if first_shape[0] == second_shape[0]:
        return first, second, None
    if first_shape[0] == 1:
        return second, first, None
    if second_shape[0] == 1:
        return first, second, None
    return None, None, "Rank-3 skip batch dimension must be one or match the data batch dimension"


def _check_skip_input(add_output, norm_output, norm_op_type: str, weight, bias=None):
    result = pattern.MatchResult()
    add = add_output.producer()
    if add is None or add.op_type != "Add":
        return result.fail(f"Input to {norm_op_type} is not produced by Add")

    data, skip, error = _skip_normalization_inputs(add)
    if error:
        return result.fail(error)

    values = [data, skip, weight]
    if bias is not None:
        values.append(bias)
    if any(value.dtype not in _SKIP_NORM_DTYPES for value in values):
        return result.fail("Fused normalization inputs must use float, float16, or bfloat16")
    if any(value.dtype != data.dtype for value in values[1:]):
        return result.fail("Fused normalization inputs must use the same dtype")

    graph = add.graph
    if graph is not None and add_output in graph.outputs:
        return result.fail("Add output is a graph output")

    norm = norm_output.producer()
    if norm.attributes.get_int("axis", -1) != -1:
        return result.fail(f"{norm_op_type} axis must be -1")
    return result


class _AddLayerNormalizationToSkipLayerNormalization(_RemoveReplacedAdd, pattern.RewriteRuleClassBase):
    def pattern(self, op, add_output, weight, bias):
        return op.LayerNormalization(
            add_output,
            weight,
            bias,
            _allow_other_attributes=True,
            _outputs=["norm_output"],
        )

    def check(self, context, add_output, weight, bias, norm_output, **_):
        return _check_skip_input(add_output, norm_output, "LayerNormalization", weight, bias)

    def rewrite(self, op, add_output, weight, bias, norm_output, **_):
        add = add_output.producer()
        data, skip, _ = _skip_normalization_inputs(add)
        epsilon = norm_output.producer().attributes.get_float("epsilon", 1e-5)
        outputs = op.SkipLayerNormalization(
            data,
            skip,
            weight,
            bias,
            _domain=MSFT_DOMAIN,
            epsilon=epsilon,
            _outputs=4,
        )
        add_output.replace_all_uses_with(outputs[3])
        self._record_replaced_add(add)
        return outputs[0]


class _AddLayerNormalizationNoBiasToSkipLayerNormalization(_RemoveReplacedAdd, pattern.RewriteRuleClassBase):
    def pattern(self, op, add_output, weight):
        return op.LayerNormalization(
            add_output,
            weight,
            _allow_other_attributes=True,
            _outputs=["norm_output"],
        )

    def check(self, context, add_output, weight, norm_output, **_):
        result = _check_skip_input(add_output, norm_output, "LayerNormalization", weight)
        if not result:
            return result
        if len(norm_output.producer().inputs) > 2:
            return result.fail("LayerNormalization has a bias")
        return result

    def rewrite(self, op, add_output, weight, norm_output, **_):
        add = add_output.producer()
        data, skip, _ = _skip_normalization_inputs(add)
        epsilon = norm_output.producer().attributes.get_float("epsilon", 1e-5)
        outputs = op.SkipLayerNormalization(
            data,
            skip,
            weight,
            _domain=MSFT_DOMAIN,
            epsilon=epsilon,
            _outputs=4,
        )
        add_output.replace_all_uses_with(outputs[3])
        self._record_replaced_add(add)
        return outputs[0]


class FuseSkipLayerNormalization(RewriteRuleSurgeon):
    """Fuse residual Add and LayerNormalization into SkipLayerNormalization."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet(
            [
                _AddLayerNormalizationToSkipLayerNormalization.rule(),
                _AddLayerNormalizationNoBiasToSkipLayerNormalization.rule(),
            ]
        )


class _AddRMSNormalizationToSkipRMSNormalization(_RemoveReplacedAdd, pattern.RewriteRuleClassBase):
    def pattern(self, op, add_output, weight):
        return op.RMSNormalization(
            add_output,
            weight,
            _allow_other_attributes=True,
            _outputs=["norm_output"],
        )

    def check(self, context, add_output, weight, norm_output, **_):
        return _check_skip_input(add_output, norm_output, "RMSNormalization", weight)

    def rewrite(self, op, add_output, weight, norm_output, **_):
        add = add_output.producer()
        data, skip, _ = _skip_normalization_inputs(add)
        epsilon = norm_output.producer().attributes.get_float("epsilon", 1e-5)
        outputs = op.SkipSimplifiedLayerNormalization(
            data,
            skip,
            weight,
            _domain=MSFT_DOMAIN,
            epsilon=epsilon,
            _outputs=4,
        )
        add_output.replace_all_uses_with(outputs[3])
        self._record_replaced_add(add)
        return outputs[0]


class FuseSkipRMSNormalization(RewriteRuleSurgeon):
    """Fuse residual Add and RMSNormalization into SkipSimplifiedLayerNormalization."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_AddRMSNormalizationToSkipRMSNormalization.rule()])
