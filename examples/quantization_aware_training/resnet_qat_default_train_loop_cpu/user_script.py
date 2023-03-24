# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class CIFAR10DataSet:
    def __init__(
        self,
        train_path="data",
        vld_path="data",
        **kwargs,
    ):
        super().__init__()
        self.train_path = train_path
        self.vld_path = vld_path
        self.setup("fit")

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
        )
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=True)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=True)


class PytorchResNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample = self.dataset[index]
        input_data = sample[0]
        label = sample[1]
        return input_data, label


def post_process(output):
    return output.softmax(1).argmax(1)


def load_pytorch_origin_model(model_path):
    print("using customized load origin model")
    resnet = torch.load(model_path)
    resnet.eval()
    return resnet


def create_benchmark_dataloader(data_dir, batchsize):
    cifar10_dataset = CIFAR10DataSet()
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batchsize, drop_last=True)
    return benchmark_dataloader


def create_train_dataloader(data_dir, batchsize):
    cifar10_dataset = CIFAR10DataSet()
    train_dataset, _ = torch.utils.data.random_split(cifar10_dataset.train_dataset, [40000, 10000])
    train_dataloader = DataLoader(PytorchResNetDataset(train_dataset), batch_size=batchsize, drop_last=True)
    return train_dataloader


def create_qat_config():
    return torch.quantization.default_qat_qconfig
