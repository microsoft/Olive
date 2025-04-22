# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path

from onnxscript.rewriter import ort_fusions

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, ir_model_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class OnnxScriptFusion(Pass):
    """Fuse Ops using onnxscript."""

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: ONNXModelHandler, config: type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        model_ir = model.load_ir_model()

        # TODO(exporter team): Different fusions support different devices
        model_ir, function_stats = ort_fusions.optimize_for_ort(model_ir)
        logger.debug("Function stats: %s", function_stats)
        # save the model to the output path and return the model
        return ir_model_to_olive_model(model_ir, output_model_path, config)
