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
    skip_special_tokens: bool = True,
) -> onnx.ModelProto:
    processor = WhisperProcessor.from_pretrained(model_name)

    # N_MELS is hardcoded to 80 but whisper-large-v3 uses 128
    # update the constant in ort-extensions
    # TODO(jambayk): Remove this once ort-extensions is updated to use dynamic N_MELS
    from onnxruntime_extensions._torch_cvt import _WhisperHParams

    original_n_mels = _WhisperHParams.N_MELS
    try:
        # can also get from config.num_mel_bins
        _WhisperHParams.N_MELS = processor.feature_extractor.feature_size

        # get pre and post processing models
        pre_model, post_model = gen_processing_models(
            processor,
            pre_kwargs={"USE_AUDIO_DECODER": use_audio_decoder, "USE_ONNX_STFT": use_onnx_stft},
            post_kwargs={"skip_special_tokens": skip_special_tokens},
            opset=opset,
        )
    finally:
        _WhisperHParams.N_MELS = original_n_mels

    # merge pre, model, and post models
    return util.quick_merge(pre_model, model, post_model)
