# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def post_process(res):
    return res.argmax(1)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    dataset = datasets.MNIST(data_dir, transform=ToTensor())
    return torch.utils.data.DataLoader(dataset, batch_size)
