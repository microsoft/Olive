# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Type

from onnxscript import ir
from onnxscript.rewriter import ort_fusions

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class OnnxScriptFusion(Pass):
    """Fuse Ops using onnxscript."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {}
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        model_proto = model.load_model()
        # ort_fusion only supports onnx ir
        model_ir = ir.from_proto(model_proto)

        # TODO(exporter team): Different fusions support different devices
        if self.accelerator_spec.execution_provider in ("CUDAExecutionProvider", "CPUExecutionProvider"):
            ort_fusions.optimize_for_ort(model_ir)

        model_proto = ir.to_proto(model_ir)
        # save the model to the output path and return the model
        return model_proto_to_olive_model(model_proto, output_model_path, config)
