# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import logging

from copy import deepcopy
from pathlib import Path
from typing import Any, Union

from olive.hardware import AcceleratorSpec
from olive.model import CompositeModelHandler, ONNXModelHandler
from olive.passes.olive_pass import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import BasePassConfig, PassConfigParam

logger = logging.getLogger(__name__)


class QairtMHA2SHA(Pass):
    """Runs QAIRT MHA to SHA transformation on ONNX model splits and saves the transformed models.

    Uses transformation API from the QAIRT SDK.
    """

    _accepts_composite_model = True

    @classmethod
    def _default_config(cls, accelerator_spec: AcceleratorSpec) -> dict[str, PassConfigParam]:
        config = {
            "mha2sha_kwargs": PassConfigParam(
                type_=dict[str, Any],
                default_value=None,
                description="Additional parameters to be passed to the MHA2SHA transformation function.",
            ),
        }
        return config

    def _run_for_config(
        self,
        model: Union[CompositeModelHandler, ONNXModelHandler],
        config: type[BasePassConfig],
        output_model_path: str,
    ) -> Union[CompositeModelHandler, ONNXModelHandler]:
        
        if isinstance(model, CompositeModelHandler):
            model_components = list(model.model_components)
            model_component_names = model.model_component_names
        elif isinstance(model, ONNXModelHandler):
            model_components = [model]
            model_component_names = ["dummy"]
        else:
            raise NotImplementedError(
                f"QairtMHA2SHA pass only supports CompositeModelHandler and ONNXModelHandler as model "
                f"components, but received type {type(model)}"
            )
        
        assert len(model_components) >= 1, (
            "There should be at least 1 component in the model."
        )
        assert all(isinstance(m, ONNXModelHandler) for m in model_components), "All components must be ONNXModelHandler"

        try:
            from qti.aisw.tools.core.utilities.framework.frameworks.onnx import OnnxModel
        except ImportError as e:
            try:
                # Backwards compatibility with older locations of OnnxModel in <= QAIRT 2.36.1
                from qti.aisw.tools.core.utilities.framework.onnx import OnnxModel
            except ImportError as e:
                logger.error(e)
                raise ImportError("Please install qti.aisw.tools and all dependencies to use QairtMHA2SHA.")
        
        new_model_components = {}
        for component, onnx_model in zip(model_component_names, model_components):
            try:
                qairt_onnx_model = OnnxModel.load(model_path=onnx_model.model_path)
                
                if hasattr(qairt_onnx_model, "mha2sha_v2"):
                    qairt_onnx_model.mha2sha_v2(**(config.mha2sha_kwargs if config.mha2sha_kwargs is not None else {}))
                else:
                    # Backwards compatibility with older definitions of OnnxModel in <= QAIRT 2.37
                    logger.warning("MHA2SHA V2 is not available for this SDK version, defaulting to MHA2SHA V1")
                    qairt_onnx_model.mha2sha(**(config.mha2sha_kwargs if config.mha2sha_kwargs is not None else {}))
                
                component_model_name = Path(onnx_model.model_path).stem
                qairt_onnx_model.export(output_model_path, prefix=component_model_name)
                new_model_components[component] = ONNXModelHandler(model_path=output_model_path, onnx_file_name=f"{component_model_name}.onnx")
            except Exception as e:
                logger.error(e)
                raise Exception(f"Failed to transform ONNX model at {onnx_model.model_path}")
    
        # Return the new model
        if isinstance(model, ONNXModelHandler):
            return ONNXModelHandler(
                model_path=output_model_path,
                onnx_file_name=model.onnx_file_name,
                model_attributes=deepcopy(onnx_model.model_attributes)
            )

        return CompositeModelHandler(
            list(new_model_components.values()),
            list(new_model_components.keys()),
            model_path=output_model_path,
            model_attributes=deepcopy(onnx_model.model_attributes)
        )