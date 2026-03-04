# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""ORT-based Cast chain elimination and com.microsoft opset fixup.

Dynamo-exported ONNX models often contain redundant Cast chains
(e.g. fp32→fp16→fp32) that double memory traffic and slow inference.
ORT has a graph optimization for this, but it is disabled by default
behind the ``session.enable_cast_chain_elimination`` session config.

Additionally, Olive ``GraphSurgeries`` may insert ``com.microsoft``
operators (such as ``LoopMHA``) without registering the custom opset
on every ONNX function scope, causing downstream failures.

This pass:
1. Ensures ``com.microsoft`` opset version 1 is declared on the model
   *and* on every ONNX function scope.
2. Runs ORT ``ORT_ENABLE_BASIC`` optimization with Cast chain
   elimination explicitly enabled to produce a cleaned-up model.
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
    models).  It then runs ORT basic graph optimization with the
    ``session.enable_cast_chain_elimination`` flag enabled to collapse
    consecutive Cast operators that cancel out.
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
                description="Run ORT basic optimization with Cast chain elimination.",
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

        # Step 2: Cast chain elimination via ORT session optimization
        if config.enable_cast_chain_elimination:
            import tempfile

            import onnxruntime as ort

            # ORT needs the patched model on disk to optimise it.
            # Large models (>2 GB) must use external data to avoid protobuf limits.
            with tempfile.TemporaryDirectory() as tmp_dir:
                tmp_path = str(Path(tmp_dir) / "model.onnx")
                onnx.save_model(
                    onnx_model,
                    tmp_path,
                    save_as_external_data=True,
                    all_tensors_to_one_file=True,
                    location="model.onnx.data",
                    convert_attribute=True,
                )

                sess_options = ort.SessionOptions()
                sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
                sess_options.optimized_model_filepath = tmp_path
                sess_options.add_session_config_entry("session.enable_cast_chain_elimination", "1")
                ort.InferenceSession(tmp_path, sess_options, providers=["CPUExecutionProvider"])

                onnx_model = onnx.load(tmp_path, load_external_data=True)

        return model_proto_to_olive_model(onnx_model, output_model_path, config)
