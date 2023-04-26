# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path

import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.model import PyTorchModel
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization
from olive.systems.local import LocalSystem


def test_openvino_quantization():
    # setup
    with tempfile.TemporaryDirectory() as tempdir:
        ov_model = get_openvino_model(tempdir)
        local_system = LocalSystem()
        data_dir = Path(tempdir) / "data"
        config = {
            "engine_config": {"device": "CPU"},
            "dataloader_func": create_dataloader,
            "data_dir": data_dir,
            "algorithms": [
                {
                    "name": "DefaultQuantization",
                    "params": {"target_device": "CPU", "preset": "performance", "stat_subset_size": 500},
                }
            ],
        }
        p = create_pass_from_dict(OpenVINOQuantization, config, disable_search=True)
        output_folder = str(Path(tempdir) / "quantized")

        # execute
        quantized_model = local_system.run_pass(p, ov_model, output_folder)

        # assert
        assert Path(quantized_model.model_path).exists()
        assert (Path(quantized_model.model_path) / "ov_model.bin").is_file()
        assert (Path(quantized_model.model_path) / "ov_model.xml").is_file()
        assert (Path(quantized_model.model_path) / "ov_model.mapping").is_file()


def get_openvino_model(tempdir):
    local_system = LocalSystem()
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tempdir)
    pytorch_model = PyTorchModel(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
    )
    openvino_conversion_config = {
        "input_shape": [1, 3, 32, 32],
    }

    p = create_pass_from_dict(OpenVINOConversion, openvino_conversion_config, disable_search=True)
    output_folder = str(Path(tempdir) / "openvino")

    # execute
    openvino_model = local_system.run_pass(p, pytorch_model, output_folder)
    return openvino_model


def create_dataloader(data_dir, batchsize):
    from addict import Dict
    from openvino.tools.pot.api import DataLoader

    class CifarDataLoader(DataLoader):
        def __init__(self, config, dataset):
            """
            Initialize config and dataset.
            :param config: created config with DATA_DIR path.
            """
            if not isinstance(config, Dict):
                config = Dict(config)
            super().__init__(config)
            self.indexes, self.pictures, self.labels = self.load_data(dataset)

        def __len__(self):
            return len(self.labels)

        def __getitem__(self, index):
            """
            Return one sample of index, label and picture.
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
            """
            Load dataset in needed format.
            :param dataset:  downloaded dataset.
            """
            pictures, labels, indexes = [], [], []

            for idx, sample in enumerate(dataset):
                pictures.append(sample[0])
                labels.append(sample[1])
                indexes.append(idx)

            return indexes, pictures, labels

    dataset_config = {"data_source": data_dir}
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    dataset = CIFAR10(root=data_dir, train=False, transform=transform, download=True)
    return CifarDataLoader(dataset_config, dataset)
