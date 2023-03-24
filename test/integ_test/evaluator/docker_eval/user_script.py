# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from torchvision import datasets
from torchvision.transforms import ToTensor


def post_process(res):
    return res.argmax(1)


def create_dataloader(data_dir, batch_size):
    dataset = datasets.MNIST(data_dir, transform=ToTensor())
    return torch.utils.data.DataLoader(dataset, batch_size)


def openvino_post_process(res):
    res = list(res.values())[0]
    return res.argmax(1)
