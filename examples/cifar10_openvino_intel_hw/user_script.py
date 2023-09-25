# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import torch
from addict import Dict
from openvino.tools.pot.api import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10


def load_pytorch_origin_model(torch_hub_model_path):
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    return torch.hub.load(torch_hub_model_path, pytorch_hub_model_name)


def post_process(result):
    return [np.argmax(result)]


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    dataset_config = {"data_source": data_dir}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    return CifarDataLoader(dataset_config, dataset)


class CifarDataLoader(DataLoader):
    def __init__(self, config, dataset):
        """Initialize config and dataset.

        :param config: created config with DATA_DIR path.
        """
        if not isinstance(config, dict):
            config = Dict(config)
        super().__init__(config)
        self.indexes, self.pictures, self.labels = self.load_data(dataset)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        """Return one sample of index, label and picture.

        :param index: index of the taken sample.
        """
        if index >= len(self):
            raise IndexError

        return (
            self.pictures[index].numpy()[
                None,
            ],
            self.labels[index],
        )

    def load_data(self, dataset):
        """Load dataset in needed format.

        :param dataset:  downloaded dataset.
        """
        pictures, labels, indexes = [], [], []

        for idx, sample in enumerate(dataset):
            pictures.append(sample[0])
            labels.append(sample[1])
            indexes.append(idx)

        return indexes, pictures, labels
