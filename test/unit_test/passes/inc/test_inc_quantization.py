# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import tempfile
from pathlib import Path

import torch
from neural_compressor.data import DefaultDataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.model import PyTorchModel
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.quantization import IncDynamicQuantization, IncQuantization, IncStaticQuantization
from olive.systems.local import LocalSystem


def test_inc_quantization():
    with tempfile.TemporaryDirectory() as tempdir:
        # setup
        ov_model = get_onnx_model(tempdir)
        local_system = LocalSystem()
        data_dir = Path(tempdir) / "data"
        config = {"data_dir": data_dir, "dataloader_func": create_dataloader}
        output_folder = str(Path(tempdir) / "quantized")

        # create IncQuantization pass
        p = IncQuantization(config, disable_search=True)
        # execute
        quantized_model = local_system.run_pass(p, ov_model, output_folder)
        # assert
        assert quantized_model.model_path.endswith(".onnx")
        assert Path(quantized_model.model_path).exists()
        assert Path(quantized_model.model_path).is_file()
        assert "QLinearConv" in [node.op_type for node in quantized_model.load_model().graph.node]

        # clean
        del p
        # create IncDynamicQuantization pass
        p = IncDynamicQuantization(config, disable_search=True)
        # execute
        quantized_model = local_system.run_pass(p, ov_model, output_folder)
        # assert
        assert quantized_model.model_path.endswith(".onnx")
        assert Path(quantized_model.model_path).exists()
        assert Path(quantized_model.model_path).is_file()
        assert "DynamicQuantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]

        # clean
        del p
        # create IncStaticQuantization pass
        p = IncStaticQuantization(config, disable_search=True)
        # execute
        quantized_model = local_system.run_pass(p, ov_model, output_folder)
        # assert
        assert quantized_model.model_path.endswith(".onnx")
        assert Path(quantized_model.model_path).exists()
        assert Path(quantized_model.model_path).is_file()
        assert "QLinearConv" in [node.op_type for node in quantized_model.load_model().graph.node]


def get_onnx_model(tempdir):
    local_system = LocalSystem()
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tempdir)
    pytorch_model = PyTorchModel(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
    )
    onnx_conversion_config = {
        "input_names": ["input"],
        "input_shapes": [[1, 3, 32, 32]],
        "output_names": ["output"],
    }

    p = OnnxConversion(onnx_conversion_config, disable_search=True)
    output_folder = str(Path(tempdir) / "onnx")

    # execute
    onnx_model = local_system.run_pass(p, pytorch_model, output_folder)
    return onnx_model


def create_dataloader(data_dir, batchsize):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    dataset = CifarDataset(CIFAR10(root=data_dir, train=False, transform=transform, download=True))
    return DefaultDataLoader(dataset=dataset, batch_size=batchsize)


class CifarDataset:
    def __init__(self, dataset):
        """
        Initialize dataset.
        :param dataset: downloaded dataset.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Return one sample of picture and label.
        :param index: index of the taken sample.
        """
        return self.dataset[index][0].numpy(), self.dataset[index][1]
