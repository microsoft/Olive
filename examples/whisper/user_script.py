# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from onnxruntime.transformers.models.t5.past_helper import PastKeyValuesHelper
from onnxruntime.transformers.models.whisper.whisper_decoder import WhisperDecoder, WhisperDecoderInputs
from onnxruntime.transformers.models.whisper.whisper_encoder_decoder_init import (
    WhisperEncoderDecoderInit,
    WhisperEncoderDecoderInitInputs,
)
from transformers import WhisperForConditionalGeneration


def get_encoder_decoder_init():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return WhisperEncoderDecoderInit(
        model,
        model,
        None,
        model.config,
        decoder_start_token_id=None,
    )


def get_decoder():
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny.en")
    return WhisperDecoder(model, None, model.config)


def get_encdec_io_config():
    model = get_encoder_decoder_init()
    use_decoder_input_ids = True

    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=use_decoder_input_ids,
        device="cpu",
        use_int32_inputs=True,
    )

    out = model(inputs.encoder_input_ids, inputs.encoder_attention_mask, inputs.decoder_input_ids)
    present = out[2]
    present_names = PastKeyValuesHelper.get_input_names(present, encoder=True)

    output_names = ["logits", "encoder_hidden_states", *present_names]

    input_names = ["encoder_input_ids", "encoder_attention_mask"]

    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference.
    # We use a workaround here: first use dim_param "1" for sequence_length, and later change to dim_value.
    sequence_length = "1"
    num_heads = str(model.config.encoder_attention_heads)
    hidden_size = str(model.config.d_model)
    head_size = str(model.config.d_model // model.config.encoder_attention_heads)
    dynamic_axes = {
        "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {
            0: "batch_size",
            1: "encode_sequence_length",
            2: hidden_size,
        },
        "logits": {
            0: "batch_size",
            1: sequence_length,
        },
    }

    if use_decoder_input_ids:
        input_names.append("decoder_input_ids")
        dynamic_axes["decoder_input_ids"] = {
            0: "batch_size",
            1: sequence_length,
        }

    for name in present_names:
        if "cross" in name:
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: "encode_sequence_length",
                3: head_size,
            }

        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                1: num_heads,
                2: sequence_length,
                3: head_size,
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
        "string_to_int_dim_params": [sequence_length, num_heads, hidden_size, head_size],
    }


def get_dec_io_config():
    # Fix past disappearing bug - duplicate first past entry
    # input_list.insert(2, input_list[2])
    model = get_decoder()
    past_names = PastKeyValuesHelper.get_past_names(model.config.decoder_layers, present=False)
    present_names = PastKeyValuesHelper.get_past_names(model.config.decoder_layers, present=True)
    present_self_names = present_names[: 2 * model.config.decoder_layers]

    input_past_names = past_names
    output_present_names = present_self_names
    output_names = ["logits", *output_present_names]

    input_names = ["input_ids"]
    input_names.append("encoder_attention_mask")
    input_names.extend(input_past_names)

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
        "encoder_attention_mask": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {0: "batch_size", 1: "encode_sequence_length / 2"},
        "logits": {0: "batch_size", 1: "sequence_length"},
    }

    for name in input_past_names:
        dynamic_axes[name] = {
            0: "batch_size",
            2: "past_decode_sequence_length" if "self" in name else "encode_sequence_length",
        }

    for name in output_present_names:
        if "cross" in name:
            dynamic_axes[name] = {0: "batch_size", 2: "encode_sequence_length"}
        else:  # self attention past state
            dynamic_axes[name] = {
                0: "batch_size",
                2: "past_decode_sequence_length + 1",
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
    }


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


class WhisperDataset:
    SAMPLE_RATE = 16000
    N_FFT = 400
    N_MELS = 80
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = N_SAMPLES // HOP_LENGTH

    def __init__(self, data_dir: str, use_audio_decoder: bool = True):
        data_dir = Path(data_dir)
        audio_files = list(data_dir.glob("*.mp3"))
        audio_files.sort(key=lambda x: x.name)
        assert len(audio_files) > 0, f"No audio files found in {data_dir}"

        self.data = []
        for audio_file in audio_files:
            if use_audio_decoder:
                with open(audio_file, "rb") as _f:
                    audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
                audio_input_name = "audio_stream"
            else:
                import librosa

                audio_blob, _ = librosa.load(audio_file)
                audio_input_name = "audio_pcm"

            audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size
            inputs = {
                audio_input_name: audio_blob,
                "max_length": np.asarray([200], dtype=np.int32),
                "min_length": np.asarray([0], dtype=np.int32),
                "num_beams": np.asarray([2], dtype=np.int32),
                "num_return_sequences": np.asarray([1], dtype=np.int32),
                "length_penalty": np.asarray([1.0], dtype=np.float32),
                "repetition_penalty": np.asarray([1.0], dtype=np.float32),
                "attention_mask": np.zeros((1, self.N_MELS, self.N_FRAMES)).astype(np.int32),
            }
            self.data.append(inputs)

        self.labels = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1
        return data, label

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]


def whisper_audio_decoder_dataloader(data_dir, batch_size=None):
    return WhisperDataset(data_dir=data_dir, use_audio_decoder=True)


def whisper_no_audio_decoder_dataloader(data_dir, batch_size=None):
    return WhisperDataset(data_dir=data_dir, use_audio_decoder=False)
