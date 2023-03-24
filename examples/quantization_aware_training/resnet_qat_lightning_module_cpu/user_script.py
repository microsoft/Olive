# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10


class PTLDataModule(LightningDataModule):
    def __init__(
        self,
        train_batch_size: int = 100,
        eval_batch_size: int = 100,
        train_path="data",
        vld_path="data",
        **kwargs,
    ):
        super().__init__()
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.train_path = train_path
        self.vld_path = vld_path
        self.setup("fit")

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
        )
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=True)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=True)

    def train_dataloader(self):
        train_set, _ = torch.utils.data.random_split(self.train_dataset, [40000, 10000])
        train_loader = DataLoader(train_set, batch_size=self.train_batch_size, shuffle=True, drop_last=True)
        return train_loader

    def val_dataloader(self):
        _, val_set = torch.utils.data.random_split(self.val_dataset, [49000, 1000])
        val_loader = DataLoader(val_set, batch_size=self.eval_batch_size, shuffle=False, drop_last=True)
        return val_loader


class PTLModule(LightningModule):
    def __init__(self, model):
        super().__init__()
        self.save_hyperparameters()
        self.model = model
        self.loss_module = torch.nn.CrossEntropyLoss()

    def forward(self, imgs):
        return self.model(imgs)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, batch, batch_idx):
        # "batch" is the output of the training data loader.
        imgs, labels = batch
        preds = self.model(imgs)
        loss = self.loss_module(preds, labels)
        acc = (preds.argmax(dim=-1) == labels).float().mean()

        # Logs the accuracy per epoch to tensorboard (weighted average over batches)
        self.log("train_acc", acc, on_step=False, on_epoch=True)
        self.log("train_loss", loss)
        return loss  # Return tensor to call ".backward" on

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        preds = self.model(imgs).argmax(dim=-1)
        acc = (labels == preds).float().mean()
        self.log("val_acc", acc)


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
    pt_data_module = PTLDataModule()
    _, val_set = torch.utils.data.random_split(pt_data_module.val_dataset, [49000, 1000])
    benchmark_dataloader = torch.utils.data.DataLoader(
        PytorchResNetDataset(val_set), batch_size=batchsize, drop_last=True
    )
    return benchmark_dataloader


def create_qat_config():
    return torch.quantization.default_qat_qconfig
