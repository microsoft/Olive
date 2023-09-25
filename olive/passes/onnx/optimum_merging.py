# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import os
from pathlib import Path
from typing import Any, Dict, Union

import onnxruntime
from onnx import ModelProto

from olive.hardware.accelerator import AcceleratorSpec
from olive.model import CompositeOnnxModel, ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config, model_proto_to_olive_model
from olive.passes.pass_config import PassConfigParam


class OptimumMerging(Pass):
    """Merges a decoder_model with its decoder_with_past_model via the Optimum library."""

    _accepts_composite_model = True

    @staticmethod
    def is_accelerator_agnostic(accelerator_spec: AcceleratorSpec) -> bool:
        """Override this method to return False by using the accelerator spec information."""
        return False

    @staticmethod
    def _default_config(accelerator_spec: AcceleratorSpec) -> Dict[str, PassConfigParam]:
        return get_external_data_config()

    def _run_for_config(
        self, model: CompositeOnnxModel, data_root: str, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        assert len(model.model_components) == 2

        # TODO(trajep): Remove this when the bug in Optimum is fixed. Optimum calls ByteSize() to see whether
        # it should be using the merged model directly or use the path instead in the model checker,
        # but unfortunately ByteSize() doesn't seem to be working correctly with external weights.
        # https://github.com/huggingface/optimum/issues/1044
        def new_byte_size_func(_):
            return 2147483648

        prev_byte_size_func = ModelProto.ByteSize
        try:
            ModelProto.ByteSize = new_byte_size_func

            from optimum.onnx import merge_decoders

            merged_model = merge_decoders(
                model.model_components[0].model_path,
                model.model_components[1].model_path,
            )
        finally:
            ModelProto.ByteSize = prev_byte_size_func

        # onnx.save will fail if the directory doesn't already exist
        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_path = os.path.join(output_model_path, "decoder_model_merged.onnx")

        olive_model = model_proto_to_olive_model(merged_model, output_model_path, config)

        # Doing a dry run of ORT allows us to remove the initializers that were orphaned by the merging step
        sess_options = onnxruntime.SessionOptions()
        sess_options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_BASIC
        sess_options.optimized_model_filepath = output_model_path

        execution_provider = self.accelerator_spec.execution_provider
        onnxruntime.InferenceSession(output_model_path, sess_options, providers=[execution_provider])

        return olive_model
