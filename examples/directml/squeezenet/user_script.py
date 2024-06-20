# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


def load_pytorch_origin_model(torch_hub_model_path):
    return torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)


class DataLoader:
    def __init__(self, batch_size):
        self.batch_size = batch_size

    def __getitem__(self, idx):
        input_data = torch.rand((self.batch_size, 3, 224, 224), dtype=torch.float16)
        label = None
        return input_data, label


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    return DataLoader(batch_size)
