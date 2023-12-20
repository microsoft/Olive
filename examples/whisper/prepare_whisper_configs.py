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
from transformers import __version__ as TransformersVersion

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
    ("gpu", "int8"): [
        "conversion",
        "transformers_optimization",
        "onnx_dynamic_quantization",
        "insert_beam_search",
        "prepost",
    ],
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
    parser.add_argument(
        "--enable_timestamps",
        action="store_true",
        help=(
            "Enable model to output timestamps along with text. Only supported in ORT >= 1.16.0 and doesn't work with"
            " whisper-large-v3. Default: False"
        ),
    )
    parser.add_argument(
        "--skip_evaluation",
        action="store_true",
        help="Skip evaluation. Default: False",
    )
    parser.add_argument(
        "--package_model",
        action="store_true",
        help=(
            "Package the final model as a zipfile along with the required onnxruntime packages and sample code."
            " Default: False"
        ),
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=1e-6,
        help=(
            "Absolute tolerance for checking float16 conversion. Only for fp16 workflow. For some cases, you can"
            " increase this value to 1e-5 or 1e-4. Default: 1e-6"
        ),
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    args = get_args(raw_args)

    # version check
    version_1_16 = version.parse(OrtVersion) >= version.parse("1.16.0")
    transformers_version_4_36 = version.parse(TransformersVersion) >= version.parse("4.36.0")

    # multi-lingual support check
    if not version_1_16:
        if args.multilingual:
            raise ValueError("Multi-lingual support is only supported in ORT >= 1.16.0")
        if args.enable_timestamps:
            raise ValueError("Enabling timestamps is only supported in ORT >= 1.16.0")
    if "large-v3" in args.model_name and args.enable_timestamps:
        print(
            "WARNING: Model has large-v3 in the name. openai/whisper-large-v3 doesn't support enabling timestamps so"
            " this might not work as expected."
        )

    # load template
    with open("whisper_template.json") as f:  # noqa: PTH123
        template_json = json.load(f)
    model_name = args.model_name

    # update model name
    template_json["input_model"]["config"]["hf_config"]["model_name"] = model_name
    if transformers_version_4_36:
        template_json["input_model"]["config"]["hf_config"]["from_pretrained_args"] = {"attn_implementation": "eager"}

    # set dataloader
    if not args.skip_evaluation:
        metric_dataloader_kwargs = template_json["evaluators"]["common_evaluator"]["metrics"][0]["user_config"][
            "func_kwargs"
        ]["dataloader_func"]
        metric_dataloader_kwargs["model_name"] = model_name
        metric_dataloader_kwargs["use_audio_decoder"] = not args.no_audio_decoder
    else:
        del template_json["evaluators"]
        template_json["engine"]["evaluator"] = None

    # update multi-lingual support
    template_json["passes"]["insert_beam_search"]["config"]["use_forced_decoder_ids"] = args.multilingual
    # update predict timestep
    template_json["passes"]["insert_beam_search"]["config"]["use_logits_processor"] = args.enable_timestamps
    # update no audio decoder
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["use_audio_decoder"] = not args.no_audio_decoder
    # update atol
    template_json["passes"]["mixed_precision"]["config"]["atol"] = args.atol

    # set model name in prepost
    template_json["passes"]["prepost"]["config"]["tool_command_args"]["model_name"] = model_name

    for device, precision in SUPPORTED_WORKFLOWS:
        workflow = SUPPORTED_WORKFLOWS[(device, precision)]
        config = deepcopy(template_json)

        # set output name
        config["engine"]["output_name"] = f"whisper_{device}_{precision}"
        # add packaging config
        if args.package_model:
            config["engine"]["packaging_config"] = {"type": "Zipfile", "name": f"whisper_{device}_{precision}"}

        # set ep
        config["engine"]["execution_providers"] = [DEVICE_TO_EP[device]]

        # set device for system
        config["systems"]["local_system"]["config"]["accelerators"] = [device]

        # add passes
        config["passes"] = {}
        for pass_name in workflow:
            pass_config = deepcopy(template_json["passes"][pass_name])
            if pass_name == "insert_beam_search":
                pass_config["config"]["fp16"] = precision == "fp16"
            if pass_name == "transformers_optimization":
                pass_config["config"]["use_gpu"] = device == "gpu"
            config["passes"][pass_name] = pass_config

        # dump config
        with open(f"whisper_{device}_{precision}.json", "w") as f:  # noqa: PTH123
            json.dump(config, f, indent=4)

    # download audio test data
    download_audio_test_data()


def download_audio_test_data():
    cur_dir = Path(__file__).parent
    data_dir = cur_dir / "data"
    data_dir.mkdir(exist_ok=True, parents=True)

    test_audio_name = "1272-141231-0002.mp3"
    test_audio_url = (
        "https://raw.githubusercontent.com/microsoft/onnxruntime-extensions/main/test/data/" + test_audio_name
    )
    test_audio_path = data_dir / test_audio_name
    if not test_audio_path.exists():
        request.urlretrieve(test_audio_url, test_audio_path)

    return test_audio_path.relative_to(cur_dir)


if __name__ == "__main__":
    main()
