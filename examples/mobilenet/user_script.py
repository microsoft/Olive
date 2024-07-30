# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from olive.data.registry import Registry
from olive.platform_sdk.qualcomm.utils.data_loader import FileListProcessedDataLoader


class MobileNetDataset(Dataset):
    def __init__(self, data_dir: str):
        data_dir = Path(data_dir)
        data_file = data_dir / "data.npy"
        self.data = torch.from_numpy(np.load(data_file))
        self.labels = None
        labels_file = data_dir / "labels.npy"
        if labels_file.exists():
            self.labels = torch.from_numpy(np.load(labels_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = torch.unsqueeze(self.data[idx], dim=0)
        label = self.labels[idx] if self.labels is not None else -1
        return {"input": data}, label


@Registry.register_dataloader()
def qnn_dataloader(dataset, data_dir: str, batch_size: int, **kwargs):
    input_list_file = "eval/input_order.txt"
    annotation_file = "eval/labels.npy"
    return FileListProcessedDataLoader(
        data_dir, batch_size=batch_size, input_list_file=input_list_file, annotations_file=annotation_file
    )


@Registry.register_dataset()
def qnn_evaluation_dataset(data_dir, **kwargs):
    return MobileNetDataset(data_dir)


@Registry.register_post_process()
def qnn_post_process(output):
    return output.argmax(axis=1)


@Registry.register_post_process()
def qnn_sdk_post_process(output):
    return np.array([output.argmax(axis=-1)])


@Registry.register_dataloader()
def mobilenet_calibration_reader(dataset, batch_size, data_dir, **kwargs):
    dataset = MobileNetDataset(data_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)
