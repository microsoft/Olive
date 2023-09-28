# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.data.config import DataConfig
from olive.data.registry import Registry
from olive.hardware import AcceleratorSpec
from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization


@pytest.mark.parametrize("data_source", ["dataloader_func", "data_config"])
def test_openvino_quantization(data_source, tmp_path):
    # setup
    ov_model = get_openvino_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "engine_config": {"device": "CPU"},
        "algorithms": [
            {
                "name": "DefaultQuantization",
                "params": {"target_device": "CPU", "preset": "performance", "stat_subset_size": 500},
            }
        ],
    }
    if data_source == "dataloader_func":
        config.update(
            {
                "dataloader_func": create_dataloader,
                "data_dir": data_dir,
            }
        )
    elif data_source == "data_config":
        config.update(
            {
                "data_config": DataConfig(
                    components={
                        "load_dataset": {
                            "name": "cifar10_dataset",
                            "type": "cifar10_dataset",
                            "params": {"data_dir": data_dir},
                        }
                    }
                )
            }
        )
    p = create_pass_from_dict(
        OpenVINOQuantization,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(ov_model, None, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert (Path(quantized_model.model_path) / "ov_model.bin").is_file()
    assert (Path(quantized_model.model_path) / "ov_model.xml").is_file()
    assert (Path(quantized_model.model_path) / "ov_model.mapping").is_file()


def get_openvino_model(tmp_path):
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tmp_path)
    pytorch_model = PyTorchModel(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
    )
    openvino_conversion_config = {
        "input_shape": [1, 3, 32, 32],
    }

    p = create_pass_from_dict(
        OpenVINOConversion,
        openvino_conversion_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "openvino")

    # execute
    return p.run(pytorch_model, None, output_folder)


@Registry.register_dataset()
def cifar10_dataset(data_dir):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    return CIFAR10(root=data_dir, train=False, transform=transform, download=True)


def create_dataloader(data_dir, batchsize, *args, **kwargs):
    from addict import Dict
    from openvino.tools.pot.api import DataLoader

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

    dataset_config = {"data_source": data_dir}
    dataset = cifar10_dataset(data_dir)
    return CifarDataLoader(dataset_config, dataset)
