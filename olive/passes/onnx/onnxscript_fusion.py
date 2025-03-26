# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path
from typing import Dict, List, Type

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import ONNXModelHandler
from olive.model.utils import resolve_onnx_path
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import BasePassConfig, PassConfigParam


class OnnxScriptFusion(Pass):
    """Fuse Ops using onnxscript.

    """

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "enable_blah_x": PassConfigParam(
                type_=bool, default_value=False, description="Whether model inputs/outputs should be left as float32"
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
        if config_dict["enable_blah_x"]:
            print("Do something")

        # fused_model = ....

        # save the model to the output path and return the model
        return model_proto_to_olive_model(fused_model.model, output_model_path, config)
