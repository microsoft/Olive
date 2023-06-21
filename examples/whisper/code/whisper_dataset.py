# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np


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
                # auto detect language and task
                "decoder_input_ids": np.asarray([[50258]], dtype=np.int32),
                # English, transcription
                # "decoder_input_ids": np.asarray([[50258, 50259, 50359, 50363]], dtype=np.int32),
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
