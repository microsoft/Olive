# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

import numpy as np
import onnx
from onnx import helper
from onnxscript import ir
from onnxscript.rewriter import RewriteRule, rewrite
from onnxscript.rewriter._basics import MatchResult

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


# TODO(anyone): Move from onnxruntime.transformers.onnx_model.OnnxModel to OnnxDAG
# or reimplement logic using onnx-rewriter
# no need to create a new instance of OnnxModel for each optimization
class ModelOptimizer:
    def __init__(self, source_model_path):
        self.source_model_path = str(source_model_path)
        self.model = onnx.load(self.source_model_path)

    def ensure_com_microsoft_opset(self):
        """Ensure com.microsoft opset v1 is declared at model and function level.

        Olive ``GraphSurgeries`` may insert ``com.microsoft`` operators (such as
        ``LoopMHA``) without registering the custom opset on every ONNX function
        scope.  This method fixes the declarations so that downstream passes and
        validators do not fail.
        """
        existing = {op.domain for op in self.model.opset_import}
        if "com.microsoft" not in existing:
            self.model.opset_import.append(helper.make_opsetid("com.microsoft", 1))
        for func in self.model.functions:
            func_domains = {op.domain for op in func.opset_import}
            if "com.microsoft" not in func_domains:
                func.opset_import.append(helper.make_opsetid("com.microsoft", 1))

    def eliminate_cast_chains(self):
        """Eliminate redundant round-trip Cast chains (e.g. fp32→fp16→fp32).

        Dynamo-exported ONNX models often contain unnecessary cast round-trips.
        This method applies a targeted onnxscript rewrite rule to collapse them
        into Identity nodes.
        """
        rules = self._get_cast_chain_rewrite_rules()
        self.model = rewrite(self.model, pattern_rewrite_rules=rules)

    @staticmethod
    def _get_cast_chain_rewrite_rules():
        """Build onnxscript rewrite rules for eliminating redundant Cast chains."""

        def _cast_cast_round_trip_pattern(op, x, to, to_ignored):
            return op.Cast(op.Cast(x, to=to_ignored), to=to)

        def _cast_cast_round_trip_check(context, x: ir.Value, to: ir.Attr, to_ignored: ir.Attr) -> MatchResult:
            check_result = MatchResult()
            if x.dtype is None:
                return check_result.fail("Input dtype unknown; cannot verify round-trip")
            if x.dtype != to.as_int():
                return check_result.fail(f"Not a round-trip cast: input dtype {x.dtype} != final cast to={to.as_int()}")
            return check_result

        def _cast_cast_round_trip_replacement(op, x, **_):
            return op.Identity(x)

        return [
            RewriteRule(
                _cast_cast_round_trip_pattern,
                _cast_cast_round_trip_replacement,
                _cast_cast_round_trip_check,
                name="CastCastRoundTrip",
            )
        ]

    def fuse_reshape_operations(self):
        # Remove unnecessary Reshape operator. Consecutive Reshape operators with latter's input being "[-1]"
        # i.e. flatten the input, the former Reshape operator is useless."""
        from onnxruntime.transformers.onnx_model import OnnxModel as TransformersOnnxModel

        o_model = TransformersOnnxModel(self.model)
        o_producers = o_model.output_name_to_node()
        o_consumers = o_model.input_name_to_nodes()

        expected_constant_value = np.array([-1])

        num_changed = 0
        for o_reshape_node in o_model.get_nodes_by_op_type("Reshape"):
            input_node_0 = o_producers.get(o_reshape_node.input[0])

            previous_is_reshape = input_node_0 and (input_node_0.op_type == "Reshape")
            current_value = o_model.get_constant_value(o_reshape_node.input[1])
            current_flattens = np.array_equal(current_value, expected_constant_value)
            only_consumer = len(o_consumers[o_reshape_node.input[0]]) == 1

            if previous_is_reshape and current_flattens and only_consumer:
                o_reshape_node.input[0] = input_node_0.input[0]
                input_node_0.input.remove(input_node_0.input[0])
                num_changed += 1

        if num_changed > 0:
            logger.debug("Fused %d redundant Reshape operators", num_changed)
            o_model.prune_graph()

    def onnxscript_optimize(self):
        try:
            import onnxscript
        except ImportError:
            logger.warning("Please install `onnxscript` to apply more optimization.")
            return
        try:
            logger.debug("Running onnxscript optimizer")
            self.model = onnxscript.optimizer.optimize(self.model)
        except Exception as e:
            if "TypeInferenceError" in str(e):
                logger.info(
                    "onnxscript optimizer failed with %s. Rerunning with shape inference disabled.",
                    str(e),
                )
                self.model = onnxscript.optimizer.optimize(self.model, onnx_shape_inference=False)
            else:
                raise

    def onnxoptimizer_optimize(self):
        try:
            from onnxoptimizer import optimize
        except ImportError:
            logger.warning("Please install `onnxoptimizer` to apply more optimization.")
            return

        self.model = optimize(self.model)


class OnnxPeepholeOptimizer(Pass):
    """Optimize ONNX model by fusing nodes.

    Runs a combination of onnxscript optimizer, onnxoptimizer, reshape
    fusion, and optionally:
    - ``com.microsoft`` opset fixup (for models that use custom ops in
      function scopes after ``GraphSurgeries``).
    - Cast chain elimination (collapses round-trip Cast chains like
      fp32→fp16→fp32 produced by dynamo export).
    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return {
            "onnxscript_optimize": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Run onnxscript optimizer for general graph optimizations.",
            ),
            "onnxoptimizer_optimize": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Run onnxoptimizer for additional graph optimizations.",
            ),
            "fuse_reshape_operations": PassConfigParam(
                type_=bool,
                default_value=True,
                description="Fuse consecutive Reshape operators where the latter flattens to [-1].",
            ),
            "fix_com_microsoft_opset": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Ensure com.microsoft opset v1 is declared on the model and all function scopes. "
                    "Enable this when GraphSurgeries inserts custom ops (e.g. LoopMHA) into function scopes."
                ),
            ),
            "cast_chain_elimination": PassConfigParam(
                type_=bool,
                default_value=False,
                description=(
                    "Apply a targeted rewrite rule to eliminate redundant round-trip Cast chains "
                    "(e.g. fp32→fp16→fp32 → identity) produced by dynamo export."
                ),
            ),
            **get_external_data_config(),
        }

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        peephole_optimizer = ModelOptimizer(model.model_path)

        if config.onnxscript_optimize:
            peephole_optimizer.onnxscript_optimize()

        if config.onnxoptimizer_optimize:
            peephole_optimizer.onnxoptimizer_optimize()

        if config.fuse_reshape_operations:
            peephole_optimizer.fuse_reshape_operations()

        if config.fix_com_microsoft_opset:
            peephole_optimizer.ensure_com_microsoft_opset()

        if config.cast_chain_elimination:
            peephole_optimizer.eliminate_cast_chains()

        # save the model to the output path and return the model
        return model_proto_to_olive_model(peephole_optimizer.model, output_model_path, config)
