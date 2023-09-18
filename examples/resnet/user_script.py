# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
import torchmetrics
from onnxruntime.quantization.calibrate import CalibrationDataReader
from pytorch_lightning import LightningDataModule, LightningModule
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.constants import Framework
from olive.model import OliveModel

# -------------------------------------------------------------------------
# Common Dataset
# -------------------------------------------------------------------------


class CIFAR10DataSet:
    def __init__(
        self,
        data_dir,
        **kwargs,
    ):
        super().__init__()
        self.train_path = data_dir
        self.vld_path = data_dir
        self.setup("fit")

    def setup(self, stage: str):
        transform = transforms.Compose(
            [transforms.Pad(4), transforms.RandomHorizontalFlip(), transforms.RandomCrop(32), transforms.ToTensor()]
        )
        self.train_dataset = CIFAR10(root=self.train_path, train=True, transform=transform, download=False)
        self.val_dataset = CIFAR10(root=self.vld_path, train=True, transform=transform, download=False)


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


# -------------------------------------------------------------------------
# Post Processing Function for Accuracy Calculation
# -------------------------------------------------------------------------


def post_process(output):
    # max_elements, max_indices = torch.max(input_tensor, dim)
    # This is a two classes classification task, result is a 2D array [[ 1.1541, -0.6622],[[-0.2137,  0.0360]]]
    # the index of this array among dimension 1 will be [1, 0], which are labels of this task
    _, preds = torch.max(output, 1)
    return preds


# -------------------------------------------------------------------------
# Dataloader for Evaluation and Performance Tuning
# -------------------------------------------------------------------------


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    cifar10_dataset = CIFAR10DataSet(data_dir)
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batch_size, drop_last=True)
    return benchmark_dataloader


# -------------------------------------------------------------------------
# Calibration Data Reader for ONNX Runtime Quantization
# -------------------------------------------------------------------------


class ResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def resnet_calibration_reader(data_dir, batch_size=16, *args, **kwargs):
    return ResnetCalibrationDataReader(data_dir, batch_size=batch_size)


# -------------------------------------------------------------------------
# Evaluation Function for Accuracy Calculation
# -------------------------------------------------------------------------


# keep this to demo/test custom evaluation function
def eval_accuracy(model: OliveModel, data_dir, batch_size, device, execution_providers):
    sess = model.prepare_session(inference_settings=None, device=device, execution_providers=execution_providers)
    dataloader = create_dataloader(data_dir, batch_size)

    preds = []
    target = []
    if model.framework == Framework.ONNX:
        input_names = [i.name for i in sess.get_inputs()]
        output_names = [o.name for o in sess.get_outputs()]
        for input_data, labels in dataloader:
            if isinstance(input_data, dict):
                input_dict = {k: input_data[k].tolist() for k in input_data.keys()}
            else:
                input_data = input_data.tolist()
                input_dict = dict(zip(input_names, [input_data]))
            res = sess.run(input_feed=input_dict, output_names=None)
            if len(output_names) == 1:
                result = torch.Tensor(res[0])
            else:
                result = torch.Tensor(res)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())

    elif model.framework == Framework.PYTORCH:
        for input_data, labels in dataloader:
            if isinstance(input_data, dict):
                result = sess(**input_data)
            else:
                result = sess(input_data)
            outputs = post_process(result)
            preds.extend(outputs.tolist())
            target.extend(labels.data.tolist())

    preds_tensor = torch.tensor(preds, dtype=torch.int)
    target_tensor = torch.tensor(target, dtype=torch.int)
    accuracy = torchmetrics.Accuracy()
    result = accuracy(preds_tensor, target_tensor)
    return result.item()


# -------------------------------------------------------------------------
# QAT config for Quantization Aware Training
# Only used by resnet_qat_default_train_loop_cpu, resnet_qat_lightning_module_cpu
# -------------------------------------------------------------------------


def create_qat_config():
    return torch.quantization.default_qat_qconfig


# -------------------------------------------------------------------------
# Training Data Loader for Quantization Aware Training using Default Training Loop
# Only used by resnet_qat_default_train_loop_cpu
# -------------------------------------------------------------------------


def create_train_dataloader(data_dir, batchsize, *args, **kwargs):
    cifar10_dataset = CIFAR10DataSet(data_dir)
    train_dataset, _ = torch.utils.data.random_split(cifar10_dataset.train_dataset, [40000, 10000])
    train_dataloader = DataLoader(PytorchResNetDataset(train_dataset), batch_size=batchsize, drop_last=True)
    return train_dataloader


# -------------------------------------------------------------------------
# Data Module for Quantization Aware Training using Lightning Module
# Only used by resnet_qat_lightning_module_cpu
# -------------------------------------------------------------------------


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
