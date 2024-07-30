# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import onnxruntime as ort
from prepare_whisper_configs import download_audio_test_data

from olive.evaluator.olive_evaluator import OnnxEvaluator
from olive.hardware import AcceleratorSpec
from olive.model import ONNXModelHandler

sys.path.append(str(Path(__file__).parent / "code"))

# ruff: noqa: T201, E402
# pylint: disable=wrong-import-position, wrong-import-order
from whisper_dataset import WhisperDataset


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Test output of Whisper Model")
    parser.add_argument("--config", type=str, required=True, help="Config used to generate model")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to audio file. If not provided, will use the test data from the config",
    )
    parser.add_argument("--language", type=str, default="english", help="Language spoken in audio")
    parser.add_argument(
        "--task",
        type=str,
        default="transcribe",
        choices=["transcribe", "translate"],
        help="whether to perform X->X speech recognition ('transcribe') or X->English translation ('translate')",
    )
    parser.add_argument(
        "--predict_timestamps",
        action="store_true",
        help="Whether to predict timestamps. Only possible for models generated with `--enable_timestamps`",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    # set ort logging level to 3 (ERROR)
    ort.set_default_logger_severity(3)

    # parse args
    args = get_args(raw_args)

    # load config
    with open(args.config) as f:
        config = json.load(f)

    # get model information
    model_name = config["input_model"]["model_components"][0]["model_path"]
    use_audio_decoder = config["passes"]["prepost"]["tool_command_args"]["use_audio_decoder"]
    # check if model is multilingual
    multilingual = config["passes"]["insert_beam_search"].get("use_forced_decoder_ids", False)
    if not multilingual and not (args.language == "english" and args.task == "transcribe"):
        print("Model is not multilingual but custom language/task provided. Will ignore custom language/task.")
    # check if model supports predicting timestamps
    timestamp_enabled = config["passes"]["insert_beam_search"].get("use_logits_processor", False)
    if args.predict_timestamps and not timestamp_enabled:
        print(
            "Model does not support predicting timestamps. Will ignore `--predict_timestamps`. Generate model with"
            " `--enable_timestamps` to support predicting timestamps."
        )
        args.predict_timestamps = False

    # get device and ep
    device = config["systems"]["local_system"]["accelerators"][0]["device"]
    ep = config["systems"]["local_system"]["accelerators"][0]["execution_providers"][0]
    accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)

    # load output model json
    output_model_json_path = Path(config["output_dir"])
    output_model_json = {}
    for model_json in output_model_json_path.glob(f"**/{config['output_name']}_{accelerator_spec}_model.json"):
        with model_json.open() as f:
            output_model_json = json.load(f)
        break

    # load output model onnx
    olive_model = ONNXModelHandler(**output_model_json["config"])

    # load audio data
    if not args.audio_path:
        args.audio_path = download_audio_test_data()

    # temporary directory for storing audio file
    temp_dir = tempfile.TemporaryDirectory()  # pylint: disable=consider-using-with
    temp_dir_path = Path(temp_dir.name)
    temp_audio_path = temp_dir_path / Path(args.audio_path).name
    shutil.copy(args.audio_path, temp_audio_path)

    # dataset
    dataset = WhisperDataset(
        data_dir=temp_dir_path,
        model_name=model_name,
        use_audio_decoder=use_audio_decoder,
        file_ext=temp_audio_path.suffix,
        language=args.language,
        task=args.task,
        predict_timestamps=args.predict_timestamps,
    )

    # create inference session
    session = olive_model.prepare_session(None, device, [ep])

    # get output
    input_data, _ = dataset[0]
    input_data = OnnxEvaluator.format_input(input_data, olive_model.io_config)
    output = olive_model.run_session(session, input_data)
    return output[0][0]


if __name__ == "__main__":
    output_text = main()
    print(output_text)
