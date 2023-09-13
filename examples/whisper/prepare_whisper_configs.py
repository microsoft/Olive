# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from copy import deepcopy
from pathlib import Path
from urllib import request

from onnxruntime import __version__ as OrtVersion
from packaging import version

SUPPORTED_WORKFLOWS = {
    ("cpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("cpu", "int8"): [
        "conversion",
        "transformers_optimization",
        "onnx_dynamic_quantization",
        "insert_beam_search",
        "prepost",
    ],
    ("cpu", "inc_int8"): [
        "conversion",
        "transformers_optimization",
        "inc_dynamic_quantization",
        "insert_beam_search",
        "prepost",
    ],
    ("gpu", "fp32"): ["conversion", "transformers_optimization", "insert_beam_search", "prepost"],
    ("gpu", "fp16"): ["conversion", "transformers_optimization", "mixed_precision", "insert_beam_search", "prepost"],
    ("gpu", "int8"): ["conversion", "onnx_dynamic_quantization", "insert_beam_search", "prepost"],
}
DEVICE_TO_EP = {
    "cpu": "CPUExecutionProvider",
    "gpu": "CUDAExecutionProvider",
}


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Prepare config file for Whisper")
    parser.add_argument("--model_name", type=str, default="openai/whisper-tiny", help="Model name")
    parser.add_argument(
        "--no_audio_decoder",
        action="store_true",
        help="Don't use audio decoder in the model. Default: False",
    )
    parser.add_argument(
        "--multilingual",
        action="store_true",
        help="Support using model for multiple languages. Only supported in ORT >= 1.16.0. Default: False",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # version check
    version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")

    # multi-lingual support check
    if args.multilingual and not version_1_16:
        raise ValueError("Multi-lingual support is only supported in ORT >= 1.16.0")

    # load template
    template_json = json.load(open("whisper_template.json", "r"))
    model_name = args.model_name

    # update model name
    template_json["input_model"]["config"]["hf_config"]["model_name"] = model_name

    # set dataloader
    template_json["evaluators"]["common_evaluator"]["metrics"][0]["user_config"]["dataloader_func"] = (
        "whisper_audio_decoder_dataloader" if not args.no_audio_decoder else "whisper_no_audio_decoder_dataloader"
    )

    # update multi-lingual support
    template_json["passes"]["insert_beam_search"]["config"]["use_forced_decoder_ids"] = args.multilingual

    # set model name in prepost
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["model_name"] = model_name

    # download audio test data
    test_audio_path = download_audio_test_data()
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["testdata_filepath"] = str(test_audio_path)

    for device, precision in SUPPORTED_WORKFLOWS:
        workflow = SUPPORTED_WORKFLOWS[(device, precision)]
        config = deepcopy(template_json)

        # set output name
        config["engine"]["output_name"] = f"whisper_{device}_{precision}"
        config["engine"]["packaging_config"]["name"] = f"whisper_{device}_{precision}"

        # set ep
        config["engine"]["execution_providers"] = [DEVICE_TO_EP[device]]

        # set device for system
        config["systems"]["local_system"]["config"]["accelerators"] = [device]

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

    # update user script
    user_script_path = Path(__file__).parent / "code" / "user_script.py"
    update_user_script(user_script_path, model_name)


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


def update_user_script(file_path, model_name):
    with open(file_path, "r") as file:
        lines = file.readlines()

    new_lines = []
    for line in lines:
        if "<model_name>" in line:
            line = line.replace("<model_name>", model_name)
        new_lines.append(line)

    with open(file_path, "w") as file:
        file.writelines(new_lines)


if __name__ == "__main__":
    main()
