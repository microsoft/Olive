# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import onnx_ir as ir


def check_scalar_constant(value: ir.Value, expected: float, name: str, *, rel_tol: float = 1e-4) -> str | None:
    if value.const_value is None:
        return f"{name} is not a constant"
    array = value.const_value.numpy()
    if array.size != 1:
        return f"{name} must contain exactly one element"
    actual = float(array.flat[0])
    if not math.isclose(actual, expected, rel_tol=rel_tol):
        return f"{name} is {actual}, expected approximately {expected}"
    return None


class ReplacedAddCleanupMixin:
    """Clean up traced Adds outside the matched pattern after rewriting."""

    _replaced_adds: list[ir.Node]

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
