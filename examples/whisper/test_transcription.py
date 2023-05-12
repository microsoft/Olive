# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
from onnxruntime_extensions import PyOrtFunction

# hard-coded audio hyperparameters
# copied from https://github.com/openai/whisper/blob/main/whisper/audio.py#L12
SAMPLE_RATE = 16000
N_FFT = 400
N_MELS = 80
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = N_SAMPLES // HOP_LENGTH


def get_args(raw_args):
    parser = argparse.ArgumentParser(description="Test output of Whisper Model")
    parser.add_argument("--config", type=str, required=True, help="Config")
    parser.add_argument(
        "--audio_path",
        type=str,
        default=None,
        help="Path to audio file. If not provided, will use the test data from the config.",
    )
    return parser.parse_args(raw_args)


def main(raw_args=None):
    # set ort logging level to 3 (ERROR)
    ort.set_default_logger_severity(3)

    # parse args
    args = get_args(raw_args)

    # load config
    config = json.load(open(args.config, "r"))

    # TODO: change the 0 to the real hardware spec when accelerator is supported
    # load output model json
    output_model_json_path = Path(config["engine"]["output_dir"]) / f"{config['engine']['output_name']}_0_model.json"
    output_model_json = json.load(open(output_model_json_path, "r"))

    # load output model onnx
    output_model_path = output_model_json["config"]["model_path"]
    model = PyOrtFunction.from_model(output_model_path)

    # load audio data
    if not args.audio_path:
        args.audio_path = Path(config["passes"]["prepost"]["config"]["tool_command_args"]["testdata_filepath"])
    use_audio_decoder = config["passes"]["prepost"]["config"]["tool_command_args"]["use_audio_decoder"]
    if use_audio_decoder:
        with open(args.audio_path, "rb") as _f:
            audio_blob = np.asarray(list(_f.read()), dtype=np.uint8)
    else:
        import librosa

        audio_blob, _ = librosa.load(args.audio_path)

    audio_blob = np.expand_dims(audio_blob, axis=0)

    output_text = model(
        audio_blob,
        np.asarray([200], dtype=np.int32),
        np.asarray([0], dtype=np.int32),
        np.asarray([2], dtype=np.int32),
        np.asarray([1], dtype=np.int32),
        np.asarray([1.0], dtype=np.float32),
        np.asarray([1.0], dtype=np.float32),
        np.zeros((1, N_MELS, N_FRAMES)).astype(np.int32),
    )
    print(output_text)


if __name__ == "__main__":
    main()
