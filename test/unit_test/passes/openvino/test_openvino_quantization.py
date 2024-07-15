# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

import pytest
import torch

from olive.data.config import DataConfig
from olive.data.registry import Registry
from olive.hardware import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization


@pytest.mark.parametrize("data_source", ["dataloader_func", "data_config"])
def test_openvino_quantization(data_source, tmp_path):
    # setup
    ov_model = get_openvino_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {}
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
                    name="test_dc_config",
                    load_dataset_config={
                        "type": "cifar10_dataset",
                        "params": {"data_dir": data_dir},
                    },
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

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


@pytest.mark.parametrize("data_source", ["dataloader_func", "data_config"])
def test_openvino_quantization_with_accuracy(data_source, tmp_path):
    # setup
    ov_model = get_openvino_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {}
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
                    name="test_dc_config",
                    load_dataset_config={
                        "type": "cifar10_dataset",
                        "params": {"data_dir": data_dir},
                    },
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

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


def get_openvino_model(tmp_path):
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tmp_path)
    pytorch_model = PyTorchModelHandler(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
    )
    openvino_conversion_config = {
        "input": [1, 3, 32, 32],
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
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    return CIFAR10(root=data_dir, train=False, transform=ToTensor(), download=True)


def create_dataloader(data_dir, batch_size, *args, **kwargs):
    from torch.utils.data.dataloader import DataLoader
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    dataset = CIFAR10(root=data_dir, train=False, transform=ToTensor(), download=True)
    return DataLoader(dataset, batch_size, shuffle=True)
