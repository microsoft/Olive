# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json

from onnxruntime.transformers.models.t5.past_helper import PastKeyValuesHelper
from onnxruntime.transformers.models.whisper.whisper_encoder_decoder_init import WhisperEncoderDecoderInitInputs

from olive.hf_utils import get_ort_whisper_for_conditional_generation


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Prepare config file for Whisper")
    parser.add_argument("--model_name", type=str, default="openai/whisper-base.en", help="Model name")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "gpu"], help="Target device")
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    whisper_model = get_ort_whisper_for_conditional_generation(args.model_name)

    # load template
    template_json = json.load(open(f"whisper_{args.device}_template.json", "r"))

    # update model name
    template_json["input_model"]["config"]["hf_config"]["model_name"] = args.model_name

    # update encoder_decoder_init io_config
    encdec_io_config = get_encdec_io_config(whisper_model.encoder_decoder_init)
    template_json["input_model"]["config"]["hf_config"]["components"][0]["io_config"] = encdec_io_config

    # update decoder io_config
    dec_io_config = get_dec_io_config(whisper_model.decoder)
    template_json["input_model"]["config"]["hf_config"]["components"][1]["io_config"] = dec_io_config

    # update model specific values for transformer optimization pass
    template_json["passes"]["transformers_optimization"]["config"][
        "num_heads"
    ] = whisper_model.config.encoder_attention_heads
    template_json["passes"]["transformers_optimization"]["config"]["hidden_size"] = whisper_model.config.d_model

    json.dump(template_json, open(f"whisper_{args.device}_config.json", "w"), indent=4)


def get_encdec_io_config(model):
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
    }


def get_dec_io_config(model):
    # Fix past disappearing bug - duplicate first past entry
    # input_list.insert(2, input_list[2])

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


if __name__ == "__main__":
    main()
