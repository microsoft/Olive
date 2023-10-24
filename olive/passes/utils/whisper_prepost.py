# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import onnx
from onnxruntime_extensions import util
from onnxruntime_extensions.cvt import gen_processing_models
from transformers import WhisperProcessor


def add_pre_post_processing_to_model(
    model: onnx.ModelProto,
    opset: int,
    model_name: str,
    use_audio_decoder: bool = True,
    use_onnx_stft: bool = True,
) -> onnx.ModelProto:
    processor = WhisperProcessor.from_pretrained(model_name)
    # get pre and post processing models
    pre_model, post_model = gen_processing_models(
        processor,
        pre_kwargs={"USE_AUDIO_DECODER": use_audio_decoder, "USE_ONNX_STFT": use_onnx_stft},
        post_kwargs={},
        opset=opset,
    )
    # merge pre, model, and post models
    return util.quick_merge(pre_model, model, post_model)
