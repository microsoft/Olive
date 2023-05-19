# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from typing import Any, Dict, Union

from olive.model import CompositeOnnxModel, ONNXModel
from olive.passes import Pass
from olive.passes.onnx.common import get_external_data_config
from olive.passes.pass_config import PassConfigParam
from pathlib import Path

from optimum.onnx import merge_decoders
import onnx
import onnxruntime
import os
from onnx import ModelProto


class OptimumMerging(Pass):
    """Merges a decoder_model with its decoder_with_past_model via the Optimum library."""

    _accepts_composite_model = True

    @staticmethod
    def _default_config() -> Dict[str, PassConfigParam]:
        config = {
            "target_provider": PassConfigParam(
                type_=str,
                default_value=None,
                description=(
                    "Target execution provider. This parameter will be removed when "
                    "accelerators/targets are visible to passes."
                ),
            ),
        }
        config.update(get_external_data_config())
        return config

    def _run_for_config(
        self, model: CompositeOnnxModel, config: Dict[str, Any], output_model_path: str
    ) -> Union[ONNXModel, CompositeOnnxModel]:
        assert len(model.model_components) == 2

        # TODO: Remove this when the bug in Optimum is fixed. Optimum calls ByteSize() to see whether
        # it should be using the merged model directly or use the path instead in the model checker,
        # but unfortunately ByteSize() doesn't seem to be working correctly with external weights.
        # https://github.com/huggingface/optimum/issues/1044
        def new_byte_size_func(_):
            return 2147483648

        prev_byte_size_func = ModelProto.ByteSize
        try:
            ModelProto.ByteSize = new_byte_size_func
            merged_model = merge_decoders(
                model.model_components[0].model_path,
                model.model_components[1].model_path,
            )
        finally:
            ModelProto.ByteSize = prev_byte_size_func

        # onnx.save will fail if the directory doesn't already exist
        Path(output_model_path).mkdir(parents=True, exist_ok=True)
        output_model_path = os.path.join(output_model_path, "decoder_model_merged.onnx")

        onnx.save(
            merged_model,
            output_model_path,
            save_as_external_data=True,
            all_tensors_to_one_file=True,
            location="decoder_model_merged.onnx.data",
        )

        # Doing a dry run of ORT allows us to remove the initializers that were orphaned by the merging step
        sess_options = onnxruntime.SessionOptions()
        sess_options.optimized_model_filepath = output_model_path

        target_provider = config["target_provider"]
        if target_provider == "directml":
            providers = ["DmlExecutionProvider"]
        elif target_provider == "cuda":
            providers = ["CUDAExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]
        
        onnxruntime.InferenceSession(output_model_path, sess_options, providers=providers)

        return ONNXModel(output_model_path, model.name)
