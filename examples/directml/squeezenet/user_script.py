# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch


def load_pytorch_origin_model(torch_hub_model_path):
    return torch.hub.load("pytorch/vision:v0.10.0", "squeezenet1_1", pretrained=True)


class DataLoader:
    def __init__(self, batchsize):
        self.batchsize = batchsize

    def __getitem__(self, idx):
        input_data = torch.rand((self.batchsize, 3, 224, 224), dtype=torch.float16)
        label = None
        return input_data, label


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    return DataLoader(batchsize)
