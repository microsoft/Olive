# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from copy import deepcopy
from pathlib import Path
from urllib import request

from onnxruntime.transformers.models.t5.past_helper import PastKeyValuesHelper
from onnxruntime.transformers.models.whisper.whisper_encoder_decoder_init import WhisperEncoderDecoderInitInputs

from olive.hf_utils import get_ort_whisper_for_conditional_generation

SUPPORTED_WORKFLOWS = {
    ("cpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("cpu", "int8"): ["conversion", "onnx_dynamic_quantization", "insert_beam_search", "prepost"],
    ("gpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("gpu", "fp16"): ["conversion", "transformers_optimization", "mixed_precision", "insert_beam_search", "prepost"],
    ("gpu", "int8"): ["conversion", "onnx_dynamic_quantization", "insert_beam_search", "prepost"],
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Prepare config file for Whisper")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny.en", help="Model name")
    parser.add_argument(
        "--no_audio_decoder",
        action="store_true",
        help="Don't use audio decoder in the model. Default: False",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    whisper_model = get_ort_whisper_for_conditional_generation(args.model_name)

    # load template
    template_json = json.load(open("whisper_template.json", "r"))

    # update model name
    template_json["input_model"]["config"]["hf_config"]["model_name"] = args.model_name

    # update encoder_decoder_init io_config
    encdec_io_config = get_encdec_io_config(whisper_model.encoder_decoder_init)
    template_json["input_model"]["config"]["hf_config"]["components"][0]["io_config"] = encdec_io_config

    # update decoder io_config
    dec_io_config = get_dec_io_config(whisper_model.decoder)
    template_json["input_model"]["config"]["hf_config"]["components"][1]["io_config"] = dec_io_config

    # set dataloader
    template_json["evaluators"]["common_evaluator"]["metrics"][0]["user_config"]["dataloader_func"] = (
        "whisper_audio_decoder_dataloader" if not args.no_audio_decoder else "whisper_no_audio_decoder_dataloader"
    )

    # update model specific values for transformer optimization pass
    template_json["passes"]["transformers_optimization"]["config"][
        "num_heads"
    ] = whisper_model.config.encoder_attention_heads
    template_json["passes"]["transformers_optimization"]["config"]["hidden_size"] = whisper_model.config.d_model

    # set model name in prepost
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["model_name"] = args.model_name

    # download audio test data
    test_audio_path = download_audio_test_data()
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["testdata_filepath"] = str(test_audio_path)

    for device, precision in SUPPORTED_WORKFLOWS:
        workflow = SUPPORTED_WORKFLOWS[(device, precision)]
        config = deepcopy(template_json)

        # set output name
        config["engine"]["output_name"] = f"whisper_{device}_{precision}"
        config["engine"]["packaging_config"]["name"] = f"whisper_{device}_{precision}"

        # set device for system
        config["systems"]["local_system"]["config"]["device"] = device

        # add passes
        config["passes"] = {}
        for pass_name in workflow:
            pass_config = deepcopy(template_json["passes"][pass_name])
            if pass_name == "transformers_optimization":
                pass_config["config"]["use_gpu"] = device == "gpu"
            if pass_name == "prepost":
                pass_config["config"]["tool_command_args"]["use_audio_decoder"] = not args.no_audio_decoder
            config["passes"][pass_name] = pass_config

        # dump config
        json.dump(config, open(f"whisper_{device}_{precision}.json", "w"), indent=4)


def download_audio_test_data():
    cur_dir = Path(__file__).parent
    data_dir = cur_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    test_audio_name = "1272-141231-0002.mp3"
    test_audio_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/" + test_audio_name
    )
    test_audio_path = data_dir / test_audio_name
    request.urlretrieve(test_audio_url, test_audio_path)

    return test_audio_path.relative_to(cur_dir)


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
        "string_to_int_dim_params": [sequence_length, num_heads, hidden_size, head_size],
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
