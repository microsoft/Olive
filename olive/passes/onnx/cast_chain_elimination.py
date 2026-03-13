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
2. Runs the ``onnxscript`` optimizer to fold or remove redundant Cast
   chains and other constant-foldable patterns.
"""

import logging
from pathlib import Path

import onnx
from onnx import helper

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


class OnnxCastChainElimination(Pass):
    """Fix com.microsoft opset declarations and eliminate redundant Cast chains.

    This pass first ensures the ``com.microsoft`` opset version 1 is
    registered on the model graph and every ONNX function scope (needed
    after ``GraphSurgeries`` inserts custom ops into dynamo-exported
    models).  It then runs the ``onnxscript`` optimizer to collapse
    consecutive Cast operators that cancel out and perform other
    peephole optimizations.
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
                description="Run onnxscript optimizer to eliminate redundant Cast chains.",
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

        # Step 2: Cast chain elimination via onnxscript optimizer
        if config.enable_cast_chain_elimination:
            import onnxscript

            try:
                onnx_model = onnxscript.optimizer.optimize(onnx_model)
            except Exception as e:
                if "TypeInferenceError" in str(e):
                    logger.info(
                        "onnxscript optimizer failed with %s. Rerunning with shape inference disabled.",
                        str(e),
                    )
                    onnx_model = onnxscript.optimizer.optimize(onnx_model, onnx_shape_inference=False)
                else:
                    raise

        return model_proto_to_olive_model(onnx_model, output_model_path, config)
