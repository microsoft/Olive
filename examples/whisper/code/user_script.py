# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from past_helper import PastKeyValuesHelper
from transformers import AutoConfig, WhisperForConditionalGeneration
from whisper_dataset import WhisperDataset
from whisper_decoder import WhisperDecoder, WhisperDecoderInputs
from whisper_encoder_decoder_init import WhisperEncoderDecoderInit, WhisperEncoderDecoderInitInputs

from olive.data.registry import Registry
from olive.model import PyTorchModelHandler


def get_encoder_decoder_init(model_path: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager")
    return WhisperEncoderDecoderInit(
        model,
        model,
        model.config,
        decoder_start_token_id=None,
    )


def get_decoder(model_path: str):
    model = WhisperForConditionalGeneration.from_pretrained(model_path, attn_implementation="eager")
    return WhisperDecoder(model, model.config)


def get_encdec_io_config(olive_model: PyTorchModelHandler):
    # model is WhisperEncoderDecoderInit
    model = olive_model.load_model()
    use_decoder_input_ids = True

    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        model.config,
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=use_decoder_input_ids,
        device="cpu",
        use_int32_inputs=True,
    )

    out = model(inputs.encoder_input_ids, inputs.decoder_input_ids)
    present = out[2]
    present_names = PastKeyValuesHelper.get_input_names(present, encoder=True)

    output_names = ["logits", "encoder_hidden_states", *present_names]

    input_names = ["encoder_input_ids"]

    # ONNX exporter might mark dimension like 'Transposepresent_value_self_1_dim_2' in shape inference.
    # We use a workaround here: first use dim_param str(model.config.encoder_attention_heads) for num_heads,
    # and later change to dim_value.
    num_heads = str(model.config.encoder_attention_heads)
    hidden_size = str(model.config.d_model)
    head_size = str(model.config.d_model // model.config.encoder_attention_heads)
    dynamic_axes = {
        "encoder_input_ids": {0: "batch_size", 1: "encode_sequence_length"},
        "encoder_hidden_states": {
            0: "batch_size",
            1: "encode_sequence_length",
            2: hidden_size,
        },
        "logits": {
            0: "batch_size",
            1: "decode_sequence_length",
        },
    }

    if use_decoder_input_ids:
        input_names.append("decoder_input_ids")
        dynamic_axes["decoder_input_ids"] = {
            0: "batch_size",
            1: "decode_sequence_length",
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
                2: "decode_sequence_length",
                3: head_size,
            }

    return {
        "input_names": input_names,
        "dynamic_axes": dynamic_axes,
        "output_names": output_names,
        "string_to_int_dim_params": [num_heads, hidden_size, head_size],
    }


def get_dec_io_config(olive_model: PyTorchModelHandler):
    # Fix past disappearing bug - duplicate first past entry
    # input_list.insert(2, input_list[2])
    config = AutoConfig.from_pretrained(olive_model.model_path)
    past_names = PastKeyValuesHelper.get_past_names(config.decoder_layers, present=False)
    present_names = PastKeyValuesHelper.get_past_names(config.decoder_layers, present=True)
    present_self_names = present_names[: 2 * config.decoder_layers]

    input_past_names = past_names
    output_present_names = present_self_names
    output_names = ["logits", *output_present_names]

    input_names = ["input_ids"]
    input_names.extend(input_past_names)

    dynamic_axes = {
        "input_ids": {0: "batch_size"},
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


def encoder_decoder_init_dummy_inputs(olive_model: PyTorchModelHandler):
    inputs = WhisperEncoderDecoderInitInputs.create_dummy(
        AutoConfig.from_pretrained(olive_model.model_path),
        batch_size=2,
        encode_sequence_length=3000,
        use_decoder_input_ids=True,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


def decoder_dummy_inputs(olive_model: PyTorchModelHandler):
    inputs = WhisperDecoderInputs.create_dummy(
        AutoConfig.from_pretrained(olive_model.model_path),
        batch_size=2,
        encode_sequence_length=3000,
        past_decode_sequence_length=5,
        device="cpu",
        use_int32_inputs=True,
    )
    return tuple(inputs.to_list())


@Registry.register_dataset()
def whisper_dataset(data_dir, **kwargs):
    model_name = kwargs["model_name"]
    use_audio_decoder = kwargs["use_audio_decoder"]
    predict_timestamps = kwargs.get("predict_timestamps", False)
    return WhisperDataset(
        data_dir=data_dir,
        model_name=model_name,
        use_audio_decoder=use_audio_decoder,
        predict_timestamps=predict_timestamps,
    )
