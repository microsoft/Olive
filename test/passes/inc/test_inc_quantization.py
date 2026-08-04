# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import platform
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from torchvision import transforms
from torchvision.datasets import CIFAR10

from olive.common.constants import OS
from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.onnx.inc_quantization import IncDynamicQuantization, IncQuantization, IncStaticQuantization


@pytest.mark.skip(reason="Dynamo export fails for MobileNetV2, need fix")
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS or torch.cuda.is_available(),
    reason="Skip test on Windows. neural-compressor import is hanging on Windows.",
)
def test_inc_quantization(tmp_path):
    ov_model = get_onnx_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "data_config": DataConfig(
            name="test_inc_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="_cifar10_val_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(type="_cifar10_val_dataloader", params={"batch_size": 1}),
        )
    }
    output_folder = str(tmp_path / "quantized")

    # create IncQuantization pass
    p = create_pass_from_dict(IncQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "QLinearConv" in [node.op_type for node in quantized_model.load_model().graph.node]

    # clean
    del p
    # create IncDynamicQuantization pass
    p = create_pass_from_dict(IncDynamicQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "DynamicQuantizeLinear" in [node.op_type for node in quantized_model.load_model().graph.node]

    # clean
    del p
    # create IncStaticQuantization pass
    p = create_pass_from_dict(IncStaticQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()
    assert "QLinearConv" in [node.op_type for node in quantized_model.load_model().graph.node]


@pytest.mark.skip(reason="Dynamo export fails for MobileNetV2, need fix")
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS, reason="Skip test on Windows. neural-compressor import is hanging on Windows."
)
def test_inc_weight_only_quantization(tmp_path):
    ov_model = get_onnx_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "approach": "weight_only",
        "data_config": DataConfig(
            name="test_inc_quant_dc_config",
            load_dataset_config=DataComponentConfig(type="_cifar10_val_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(type="_cifar10_val_dataloader"),
        ),
    }
    output_folder = str(tmp_path / "quantized")

    # create IncQuantization pass
    p = create_pass_from_dict(IncQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()

    # clean
    del p
    # create IncStaticQuantization pass
    p = create_pass_from_dict(IncStaticQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()


@pytest.mark.skip(reason="Dynamo export fails for MobileNetV2, need fix")
@pytest.mark.skipif(
    platform.system() == OS.WINDOWS, reason="Skip test on Windows. neural-compressor import is hanging on Windows."
)
@patch.dict("neural_compressor.quantization.STRATEGIES", {"auto": MagicMock()})
@patch("olive.passes.onnx.inc_quantization.model_proto_to_olive_model")
def test_inc_quantization_with_data_config(mock_model_saver, tmp_path):
    ov_model = get_onnx_model(tmp_path)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {"approach": "static", "data_dir": data_dir, "data_config": DataConfig(name="test_dc_config")}
    output_folder = str(tmp_path / "quantized")

    mock_model_saver.return_value = ov_model
    # create IncQuantization pass
    p = create_pass_from_dict(IncQuantization, config, disable_search=True)
    # execute
    quantized_model = p.run(ov_model, output_folder)
    # assert
    assert quantized_model.model_path.endswith(".onnx")
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()


def get_onnx_model(tmp_path):
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tmp_path)
    pytorch_model = PyTorchModelHandler(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
        io_config={"input_names": ["input"], "input_shapes": [[1, 3, 32, 32]], "output_names": ["output"]},
    )
    onnx_conversion_config = {}

    p = create_pass_from_dict(OnnxConversion, onnx_conversion_config, disable_search=True)
    output_folder = str(tmp_path / "onnx")

    # execute
    return p.run(pytorch_model, output_folder)


@Registry.register_dataset()
def _cifar10_val_dataset(data_dir, **kwargs):
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261))]
    )
    return CIFAR10(root=data_dir, train=False, transform=transform, download=True)


@Registry.register_dataloader()
def _cifar10_val_dataloader(dataset, batch_size, **kwargs):
    # import neural_compressor here to avoid hanging on Windows
    from neural_compressor.data import DefaultDataLoader

    return DefaultDataLoader(dataset=CifarDataset(dataset), batch_size=batch_size)


class CifarDataset:
    def __init__(self, dataset):
        """Initialize dataset.

        :param dataset: downloaded dataset.
        """
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """Return one sample of picture and label.

        :param index: index of the taken sample.
        """
        return self.dataset[index][0].numpy(), self.dataset[index][1]
