# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset

from olive.platform_sdk.qualcomm.utils.data_loader import FileListProcessedDataLoader


class MobileNetDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        data_file = data_dir / "data.npy"
        self.data = np.load(data_file)
        self.labels = None
        labels_file = data_dir / "labels.npy"
        if labels_file.exists():
            self.labels = np.load(labels_file)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.labels[idx] if self.labels is not None else -1
        return {"input": data}, label


class MobileNetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 1):
        super().__init__()
        self.dataset = MobileNetDataset(data_dir)
        self.dataloader = DataLoader(self.dataset, batch_size=batch_size, shuffle=False)
        self.iter = iter(self.dataloader)

    def get_next(self):
        if self.iter is None:
            self.iter = iter(self.dataloader)
        try:
            batch = next(self.iter)[0]
        except StopIteration:
            return None

        return {k: v.detach().cpu().numpy() for k, v in batch.items()}

    def rewind(self):
        self.iter = None


def qnn_data_loader(data_dir: str, batch_size: int, *args, **kwargs):
    input_list_file = "eval/input_order.txt"
    annotation_file = "eval/labels.npy"
    return FileListProcessedDataLoader(
        data_dir, batch_size=batch_size, input_list_file=input_list_file, annotations_file=annotation_file
    )


def evaluation_dataloader(data_dir, batch_size, *args, **kwargs):
    dataset = MobileNetDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size)


def post_process(output):
    return output.argmax(axis=1)


def qnn_sdk_post_process(output):
    return np.array([output.argmax(axis=-1)])


def mobilenet_calibration_reader(data_dir, batch_size, *args, **kwargs):
    return MobileNetCalibrationDataReader(data_dir, batch_size=batch_size)
