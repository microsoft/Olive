# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from onnxruntime.transformers.models.whisper.whisper_decoder import WhisperDecoderInputs
from onnxruntime.transformers.models.whisper.whisper_encoder_decoder_init import WhisperEncoderDecoderInitInputs


def encoder_decoder_init_dummy_inputs(model):
    model = model.load_model()
    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=True,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def decoder_dummy_inputs(model):
    model = model.load_model()
    inputs = WhisperDecoderInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        past_decode_sequence_length=5,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())
