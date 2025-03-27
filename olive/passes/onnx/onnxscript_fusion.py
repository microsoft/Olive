# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, Type

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
        config = {
            "device": PassConfigParam(
                type_=str,
                description=(
                    "The device to use for conversion, e.g., 'cuda' or 'cpu'. If not specified, will use 'cpu' for"
                    " PyTorch model and 'cuda' for DistributedHfModel."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: ONNXModelHandler, config: Type[BasePassConfig], output_model_path: str
    ) -> ONNXModelHandler:
        from onnxruntime.transformers.onnx_model import OnnxModel

        output_model_path = resolve_onnx_path(output_model_path, Path(model.model_path).name)

        onnx_model = OnnxModel(model.load_model())
        config_dict = config.dict()

        # TODO(titaiwang): Different fusions support different devices
        if config_dict["decive"] in ("cuda", "cpu"):
            ort_fusions.optimize_for_ort(onnx_model)

        # save the model to the output path and return the model
        return model_proto_to_olive_model(onnx_model.model, output_model_path, config)
