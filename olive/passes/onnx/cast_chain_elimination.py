# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Cast chain elimination and com.microsoft opset fixup.

Dynamo-exported ONNX models often contain redundant Cast chains
(e.g. fp32→fp16→fp32) that double memory traffic and slow inference.

Additionally, Olive ``GraphSurgeries`` may insert ``com.microsoft``
operators (such as ``LoopMHA``) without registering the custom opset
on every ONNX function scope, causing downstream failures.

This pass:
1. Ensures ``com.microsoft`` opset version 1 is declared on the model
   *and* on every ONNX function scope.
2. Applies targeted onnxscript rewrite rules to eliminate redundant
   round-trip Cast chains (e.g. fp32→fp16→fp32 → identity).
"""

import logging
from pathlib import Path

import onnx
from onnx import helper
from onnxscript import ir
from onnxscript.rewriter import RewriteRuleClassBase, rewrite
from onnxscript.rewriter._basics import MatchResult

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


def _ensure_com_microsoft_opset(model: onnx.ModelProto):
    """Ensure com.microsoft opset v1 is declared at model and function level."""
    existing = {op.domain for op in model.opset_import}
    if "com.microsoft" not in existing:
        model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
    for func in model.functions:
        func_domains = {op.domain for op in func.opset_import}
        if "com.microsoft" not in func_domains:
            func.opset_import.append(helper.make_opsetid("com.microsoft", 1))


def _get_cast_chain_rewrite_rules():
    """Build onnxscript rewrite rules for eliminating redundant Cast chains.

    Returns a list of ``RewriteRule`` instances that target round-trip
    Cast patterns (e.g. fp32→fp16→fp32) produced by dynamo export.
    """

    class _CastCastRoundTrip(RewriteRuleClassBase):
        """Collapse ``Cast(Cast(x, to=T2), to=T3)`` to ``Identity(x)`` when T3 matches x's type.

        Dynamo-exported models frequently insert unnecessary cast round-trips
        (e.g. fp32→fp16→fp32).  When the final cast type equals the original
        input type the entire chain is a no-op and can be replaced by Identity.
        """

        def pattern(self, op, x, to, to_ignored):
            return op.Cast(op.Cast(x, to=to_ignored), to=to)

        def check(self, context, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr) -> MatchResult:
            check_result = MatchResult()
            if x.dtype is None:
                return check_result.fail("Input dtype unknown; cannot verify round-trip")
            if x.dtype != to.as_int():
                return check_result.fail(f"Not a round-trip cast: input dtype {x.dtype} != final cast to={to.as_int()}")
            return check_result

        def rewrite(self, op, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr):
            return op.Identity(x)

    return [_CastCastRoundTrip().rule()]


class OnnxCastChainElimination(Pass):
    """Fix com.microsoft opset declarations and eliminate redundant Cast chains.

    This pass first ensures the ``com.microsoft`` opset version 1 is
    registered on the model graph and every ONNX function scope (needed
    after ``GraphSurgeries`` inserts custom ops into dynamo-exported
    models).  It then applies targeted onnxscript rewrite rules to
    collapse consecutive Cast operators that form a round-trip
    (e.g. fp32→fp16→fp32 → identity).
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "fix_opset": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Ensure com.microsoft opset v1 is declared on all scopes.",
            ),
            "enable_cast_chain_elimination": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Apply rewrite rules to eliminate redundant round-trip Cast chains.",
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        onnx_model = model.load_model()

        # Step 1: Opset fixup
        if config.fix_opset:
            _ensure_com_microsoft_opset(onnx_model)

        # Step 2: Cast chain elimination via targeted onnxscript rewrite rules
        if config.enable_cast_chain_elimination:
            rules = _get_cast_chain_rewrite_rules()
            onnx_model = rewrite(onnx_model, pattern_rewrite_rules=rules)

        return model_proto_to_olive_model(onnx_model, output_model_path, config)
