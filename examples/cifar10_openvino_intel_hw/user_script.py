# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import numpy as np
import torch
from openvino import CompiledModel
from sklearn.metrics import accuracy_score
from torch.utils.data.dataloader import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import ToTensor


def load_pytorch_origin_model(torch_hub_model_path):
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    return torch.hub.load(torch_hub_model_path, pytorch_hub_model_name, trust_repo=True)


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    dataset = CIFAR10(root=data_dir, train=False, transform=ToTensor(), download=True)
    return DataLoader(dataset, batchsize, shuffle=True)


def validate(model: CompiledModel, validation_loader) -> float:
    predictions = []
    references = []

    output = model.outputs[0]

    for data_item, target in validation_loader:
        pred = model(data_item)[output]
        predictions.append(np.argmax(pred, axis=1))
        references.append(target)

    predictions = np.concatenate(predictions, axis=0)
    references = np.concatenate(references, axis=0)
    return accuracy_score(predictions, references)
