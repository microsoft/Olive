# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import torch
from onnxruntime.quantization.calibrate import CalibrationDataReader
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.constants import Framework
from olive.evaluator.accuracy import AccuracyScore
from olive.model import OliveModel


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


def create_dataloader(data_dir, batch_size):
    cifar10_dataset = CIFAR10DataSet(data_dir)
    _, val_set = torch.utils.data.random_split(cifar10_dataset.val_dataset, [49000, 1000])
    benchmark_dataloader = DataLoader(PytorchResNetDataset(val_set), batch_size=batch_size, drop_last=True)
    return benchmark_dataloader


def post_process(output):
    # max_elements, max_indices = torch.max(input_tensor, dim)
    # This is a two classes classification task, result is a 2D array [[ 1.1541, -0.6622],[[-0.2137,  0.0360]]]
    # the index of this array among dimension 1 will be [1, 0], which are labels of this task
    _, preds = torch.max(output, 1)
    return preds


class ResnetCalibrationDataReader(CalibrationDataReader):
    def __init__(self, data_dir: str, batch_size: int = 16):
        super().__init__()
        self.iterator = iter(create_dataloader(data_dir, batch_size))

    def get_next(self) -> dict:
        try:
            return {"input": next(self.iterator)[0].numpy()}
        except Exception:
            return None


def resnet_calibration_reader(data_dir, batch_size=16):
    return ResnetCalibrationDataReader(data_dir, batch_size=batch_size)


# keep this to demo/test custom evaluation function
def eval_accuracy(model: OliveModel, data_dir, batch_size, device):
    sess = model.prepare_session(inference_settings=None, device=device)
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

    return AccuracyScore().measure(preds, target)
