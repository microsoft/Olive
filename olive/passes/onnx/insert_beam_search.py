# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict

import numpy as np
import onnx

from olive.model import CompositeOnnxModel, ONNXModel, OliveModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam

class InsertBeamSearchPass(Pass):
    """Insert Beam Search Op."""

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        return {}

    def _run_for_config(self, model: OliveModel, config: Dict[str, Any], output_model_path: str) -> ONNXModel:
        if isinstance(model, ONNXModel):
            return model
        
        if not isinstance(model, CompositeOnnxModel):
            raise ValueError
        
        first_m = model.component[0]
        second_m = model.component[1]

        combined_model = insert_beamsearch_op(first_m, second_m)

        # save the model to the output path and return the model
        output_model_path = ONNXModel.resolve_path(output_model_path)
        return model_proto_to_olive_model(combined_model.model, output_model_path, config, model.name)
