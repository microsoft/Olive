# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Union

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeOnnxModel, ONNXModel, OptimumModel
from olive.model.hf_utils import HFConfig
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam


class OptimumConversion(Pass):
    """Convert a Optimum model to ONNX model using the Optimum export function."""

    _requires_user_script = True

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        config = {
            "target_opset": PassConfigParam(
                type_=int, default_value=14, description="The version of the default (ai.onnx) opset to target."
            )
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: OptimumModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        assert len(model.model_components) > 0

        from optimum.exporters.onnx import main_export as export_optimum_model

        # TODO: export into temp dir and then move to sub-dirs of output_model_path
        # so that we only keep the final model files in the output_model_path
        # and track external data if present
        hf_config = deepcopy(model.hf_config) or HFConfig()
        export_optimum_model(
            model.model_path or hf_config.model_name,
            output_model_path,
            opset=config["target_opset"],
            no_post_process=True,
        )

        onnx_model_components = [
            ONNXModel(str(Path(output_model_path) / model_component), model_attributes=model.model_attributes)
            for model_component in model.model_components
        ]
        onnx_model_component_names = [Path(model_component).stem for model_component in model.model_components]

        if len(onnx_model_components) == 1:
            return ONNXModel(Path(output_model_path) / model.model_components[0])

        return CompositeOnnxModel(onnx_model_components, onnx_model_component_names)
