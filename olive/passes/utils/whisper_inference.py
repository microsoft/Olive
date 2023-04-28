# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
import os
import pprint

import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime
from onnxruntime_extensions import get_library_path
from whisper_prepost import add_pre_post_processing_to_model


def _load_test_data(filepath: str) -> npt.NDArray[np.uint8]:
    with open(filepath, "rb") as strm:
        audio_blob = np.asarray(list(strm.read()), dtype=np.uint8)
    audio_blob = np.expand_dims(audio_blob, axis=0)  # add a batch_size
    return audio_blob


def _main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--name",
        dest="name",
        required=True,
        type=str,
        help="Name of the whisper model",
    )
    parser.add_argument(
        "--input-filepath",
        dest="input_filepath",
        required=True,
        type=str,
        help="Input whisper model filepath",
    )
    parser.add_argument(
        "--output-filepath",
        dest="output_filepath",
        required=True,
        type=str,
        help="Output whisper model filepath",
    )
    parser.add_argument(
        "--testdata-filepath",
        dest="testdata_filepath",
        required=True,
        type=str,
        help="Test data filepath",
    )
    args = parser.parse_args()

    if not os.path.isfile(args.output_filepath):
        input_model = onnx.load(args.input_filepath)
        add_pre_post_processing_to_model(args.name, input_model, args.output_filepath, args.testdata_filepath)

    audio_blob = _load_test_data(args.testdata_filepath)

    SAMPLE_RATE = 16000
    N_MELS = 80
    HOP_LENGTH = 160
    CHUNK_LENGTH = 30
    N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
    N_FRAMES = N_SAMPLES // HOP_LENGTH

    so = onnxruntime.SessionOptions()
    so.register_custom_ops_library(get_library_path())

    session = onnxruntime.InferenceSession(args.output_filepath, so, providers=["CPUExecutionProvider"])
    inputs = {
        "audio_stream": audio_blob,
        "max_length": np.asarray([200], dtype=np.int32),
        "min_length": np.asarray([0], dtype=np.int32),
        "num_beams": np.asarray([2], dtype=np.int32),
        "num_return_sequences": np.asarray([1], dtype=np.int32),
        "length_penalty": np.asarray([1.0], dtype=np.float32),
        "repetition_penalty": np.asarray([1.0], dtype=np.float32),
        "attention_mask": np.zeros((1, N_MELS, N_FRAMES)).astype(np.int32),
    }
    outputs = session.run(None, inputs)[0]

    # from onnxruntime_extensions import PyOrtFunction
    #
    # m_final = PyOrtFunction.from_model(args.output_filepath, cpu_only=True)
    # outputs = m_final(
    #     audio_blob,
    #     np.asarray([200], dtype=np.int32),
    #     np.asarray([0], dtype=np.int32),
    #     np.asarray([2], dtype=np.int32),
    #     np.asarray([1], dtype=np.int32),
    #     np.asarray([1.0], dtype=np.float32), np.asarray([1.0], dtype=np.float32),
    #     np.zeros((1, N_MELS, N_FRAMES)).astype(np.int32))

    pprint.pprint(outputs[0])
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(_main())
