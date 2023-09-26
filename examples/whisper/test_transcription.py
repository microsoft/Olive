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
from olive.model import ONNXModel

sys.path.append(str(Path(__file__).parent / "code"))

from whisper_dataset import WhisperDataset  # noqa: E402


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
    return parser.parse_args(raw_args)


def main(raw_args=None):
    # set ort logging level to 3 (ERROR)
    ort.set_default_logger_severity(3)

    # parse args
    args = get_args(raw_args)

    # load config
    with open(args.config) as f:  # noqa: PTH123
        config = json.load(f)

    # get model information
    use_audio_decoder = config["passes"]["prepost"]["config"]["tool_command_args"]["use_audio_decoder"]
    multilingual = config["passes"]["insert_beam_search"]["config"].get("use_forced_decoder_ids", False)
    if not multilingual and not (args.language == "english" and args.task == "transcribe"):
        print("Model is not multilingual but custom language/task provided. Will ignore custom language/task.")

    # get device and ep
    device = config["systems"]["local_system"]["config"]["accelerators"][0]
    ep = config["engine"]["execution_providers"][0]
    accelerator_spec = AcceleratorSpec(accelerator_type=device, execution_provider=ep)

    # load output model json
    output_model_json_path = Path(config["engine"]["output_dir"])
    for model_json in output_model_json_path.glob(
        f"**/{config['engine']['output_name']}_{accelerator_spec}_model.json"
    ):
        with model_json.open() as f:
            output_model_json = json.load(f)
        break

    # load output model onnx
    olive_model = ONNXModel(**output_model_json["config"])

    # load audio data
    if not args.audio_path:
        args.audio_path = download_audio_test_data()

    # temporary directory for storing audio file
    temp_dir = tempfile.TemporaryDirectory()
    temp_dir_path = Path(temp_dir.name)
    temp_audio_path = temp_dir_path / Path(args.audio_path).name
    shutil.copy(args.audio_path, temp_audio_path)

    # dataset
    dataset = WhisperDataset(
        data_dir=temp_dir_path,
        use_audio_decoder=use_audio_decoder,
        file_ext=temp_audio_path.suffix,
        language=args.language,
        task=args.task,
    )

    # create inference session
    session = olive_model.prepare_session(None, device, [ep])

    # get output
    input_data, _ = dataset[0]
    input_data = OnnxEvaluator.format_input(input_data, olive_model.get_io_config())
    output = session.run(None, input_data)
    return output[0][0]


if __name__ == "__main__":
    output_text = main()
    print(output_text)
