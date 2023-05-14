# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, Union

from olive.common.utils import tensor_data_to_device
from olive.model import CompositeOnnxModel, ONNXModel, OptimumModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam
from pathlib import Path

from optimum.exporters.onnx import main_export as export_optimum_model


class OptimumConversion(Pass):
    """Convert a Optimum model to ONNX model using the Optimum export function."""

    _requires_user_script = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            )
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: OptimumModel, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        assert len(model.model_components) > 0

        input("LALALALA")

#        export_optimum_model(
#            model.model_path,
#            output_model_path,
#            opset=config["target_opset"],
#            no_post_process=True,
#        )

        onnx_model_components = [
            ONNXModel(str(Path(output_model_path) / model_component), model.name)
            for model_component in model.model_components
        ]

        if len(onnx_model_components) == 1:
            return ONNXModel(output_model_path / model.model_components[0], model.name)
        
        return CompositeOnnxModel(onnx_model_components, model.name)
