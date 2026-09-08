# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import TYPE_CHECKING

from onnxscript.rewriter import pattern

from olive.constants import MSFT_DOMAIN
from olive.passes.onnx.graph_surgery._common import ReplacedAddCleanupMixin, check_scalar_constant
from olive.passes.onnx.graph_surgery.base import RewriteRuleSurgeon

if TYPE_CHECKING:
    import onnx_ir as ir

# ONNXScript binds each rule's named pattern operands to its callbacks.
# pylint: disable=arguments-differ

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFFICIENT = 0.044715


class _ExactGelu(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, sqrt_2, one, half):
        divided = op.Div(x, sqrt_2)
        erf = op.Erf(divided)
        shifted = op.Add(erf, one)
        scaled = op.Mul(x, shifted)
        return op.Mul(scaled, half)

    def check(self, context, sqrt_2, one, half, **_):
        result = pattern.MatchResult()
        for value, expected, name in (
            (sqrt_2, _SQRT_2, "sqrt(2) divisor"),
            (one, 1.0, "Add constant"),
            (half, 0.5, "Mul half constant"),
        ):
            if error := check_scalar_constant(value, expected, name, rel_tol=1e-3):
                return result.fail(error)
        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="none")


class _ExactGeluHalfFirst(_ExactGelu):
    def pattern(self, op, x, sqrt_2, one, half):
        divided = op.Div(x, sqrt_2)
        erf = op.Erf(divided)
        shifted = op.Add(erf, one)
        scaled = op.Mul(x, shifted)
        return op.Mul(half, scaled)


class _ApproximateGelu(pattern.RewriteRuleClassBase):
    def pattern(self, op, x, three, coefficient, sqrt_2_over_pi, one, half):
        cubed = op.Pow(x, three)
        scaled_cube = op.Mul(coefficient, cubed)
        inner = op.Add(x, scaled_cube)
        scaled = op.Mul(sqrt_2_over_pi, inner)
        tanh = op.Tanh(scaled)
        shifted = op.Add(tanh, one)
        activated = op.Mul(x, shifted)
        return op.Mul(activated, half)

    def check(self, context, three, coefficient, sqrt_2_over_pi, one, half, **_):
        result = pattern.MatchResult()
        for value, expected, name in (
            (three, 3.0, "Pow exponent"),
            (coefficient, _GELU_COEFFICIENT, "Gelu coefficient"),
            (sqrt_2_over_pi, _SQRT_2_OVER_PI, "sqrt(2/pi) constant"),
            (one, 1.0, "Add constant"),
            (half, 0.5, "Mul half constant"),
        ):
            if error := check_scalar_constant(value, expected, name, rel_tol=1e-3):
                return result.fail(error)
        return result

    def rewrite(self, op, x, **_):
        return op.Gelu(x, approximate="tanh")


class _ApproximateGeluHalfFirst(_ApproximateGelu):
    def pattern(self, op, x, three, coefficient, sqrt_2_over_pi, one, half):
        cubed = op.Pow(x, three)
        scaled_cube = op.Mul(coefficient, cubed)
        inner = op.Add(x, scaled_cube)
        scaled = op.Mul(sqrt_2_over_pi, inner)
        tanh = op.Tanh(scaled)
        shifted = op.Add(tanh, one)
        activated = op.Mul(x, shifted)
        return op.Mul(half, activated)


class FuseGelu(RewriteRuleSurgeon):
    """Fuse exact and tanh-approximate decomposed Gelu subgraphs into ONNX Gelu."""

    def call_ir(self, model: ir.Model) -> ir.Model:
        if model.opset_imports.get("", 0) < 20:
            return model
        return super().call_ir(model)

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet(
            [
                _ExactGelu.rule(),
                _ExactGeluHalfFirst.rule(),
                _ApproximateGelu.rule(),
                _ApproximateGeluHalfFirst.rule(),
            ]
        )


def _bias_gelu_inputs(add):
    shapes = [value.shape for value in add.inputs]
    if any(shape is None for shape in shapes):
        return None, None, "Add input shapes must be known"

    bias_indices = [index for index, shape in enumerate(shapes) if len(shape) == 1]
    if len(bias_indices) == 2:
        data_index, bias_index = 0, 1
    elif len(bias_indices) == 1:
        bias_index = bias_indices[0]
        data_index = 1 - bias_index
    else:
        return None, None, "Add must have a rank-1 bias input"

    data_shape = shapes[data_index]
    bias_shape = shapes[bias_index]
    if len(data_shape) < 1:
        return None, None, "BiasGelu data input rank must be at least one"

    hidden_size = data_shape[-1]
    bias_size = bias_shape[0]
    if not isinstance(hidden_size, int) or not isinstance(bias_size, int):
        return None, None, "Bias and data last dimensions must be statically known"
    if hidden_size != bias_size:
        return None, None, f"Bias size {bias_size} does not match data last dimension {hidden_size}"

    return add.inputs[data_index], add.inputs[bias_index], None


class _AddGeluToBiasGelu(ReplacedAddCleanupMixin, pattern.RewriteRuleClassBase):
    def pattern(self, op, add_output):
        return op.Gelu(add_output, _outputs=["gelu_output"])

    def check(self, context, add_output, gelu_output, **_):
        result = pattern.MatchResult()

        gelu = gelu_output.producer()
        approximate = gelu.attributes.get("approximate", None)
        approximate_value = approximate.value if approximate is not None else "none"
        if approximate_value != "none":
            return result.fail(f"Gelu uses approximate='{approximate_value}', BiasGelu requires exact Gelu")

        add = add_output.producer()
        if add is None or add.op_type != "Add":
            return result.fail("Input to Gelu is not produced by Add")

        uses = list(add_output.uses())
        if len(uses) != 1:
            return result.fail(f"Add output has {len(uses)} consumers, expected exactly one")

        _, _, error = _bias_gelu_inputs(add)
        if error:
            return result.fail(error)

        return result

    def rewrite(self, op, add_output, **_):
        add = add_output.producer()
        data, bias, _ = _bias_gelu_inputs(add)
        self._record_replaced_add(add)
        return op.BiasGelu(data, bias, _domain=MSFT_DOMAIN)


class FuseBiasGelu(RewriteRuleSurgeon):
    """Fuse a compatible bias Add followed by exact Gelu into BiasGelu."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_AddGeluToBiasGelu.rule()])
