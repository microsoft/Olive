# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math

from onnxscript.rewriter import pattern

from olive.constants import MSFT_DOMAIN
from olive.passes.onnx.graph_surgery.base import RewriteRuleSurgeon

_SQRT_2 = math.sqrt(2.0)
_SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)
_GELU_COEFFICIENT = 0.044715


def _check_constant(value, expected: float, name: str) -> str | None:
    if value.const_value is None:
        return f"{name} is not a constant"
    actual = float(value.const_value.numpy().flat[0])
    if not math.isclose(actual, expected, rel_tol=1e-3):
        return f"{name} is {actual}, expected approximately {expected}"
    return None


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
            if error := _check_constant(value, expected, name):
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
            if error := _check_constant(value, expected, name):
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

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet(
            [
                _ExactGelu.rule(),
                _ExactGeluHalfFirst.rule(),
                _ApproximateGelu.rule(),
                _ApproximateGeluHalfFirst.rule(),
            ]
        )


class _AddGeluToBiasGelu(_RemoveReplacedAdd, pattern.RewriteRuleClassBase):
    def pattern(self, op, add_output):
        return op.Gelu(add_output, _outputs=["gelu_output"])

    def check(self, context, add_output, gelu_output, **_):
        result = pattern.MatchResult()

        gelu = gelu_output.producer()
        approximate = gelu.attributes.get("approximate", None)
        approximate_value = approximate.value if approximate is not None else "none"
        if approximate_value != "tanh":
            return result.fail(f"Gelu uses approximate='{approximate_value}', BiasGelu requires 'tanh'")

        add = add_output.producer()
        if add is None or add.op_type != "Add":
            return result.fail("Input to Gelu is not produced by Add")

        uses = list(add_output.uses())
        if len(uses) != 1:
            return result.fail(f"Add output has {len(uses)} consumers, expected exactly one")

        return result

    def rewrite(self, op, add_output, **_):
        add = add_output.producer()
        self._record_replaced_add(add)
        return op.BiasGelu(add.inputs[0], add.inputs[1], _domain=MSFT_DOMAIN)


class FuseBiasGelu(RewriteRuleSurgeon):
    """Fuse a single-use Add followed by tanh-approximate Gelu into BiasGelu."""

    def rules(self) -> pattern.RewriteRuleSet:
        return pattern.RewriteRuleSet([_AddGeluToBiasGelu.rule()])
