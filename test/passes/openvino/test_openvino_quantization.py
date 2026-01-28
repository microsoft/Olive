# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import shutil
from pathlib import Path

import pytest
import torch

from olive.data.config import DataComponentConfig, DataConfig
from olive.data.registry import Registry
from olive.hardware import AcceleratorSpec
from olive.model import PyTorchModelHandler
from olive.passes.olive_pass import create_pass_from_dict
from olive.passes.onnx.conversion import OnnxConversion
from olive.passes.openvino.conversion import OpenVINOConversion
from olive.passes.openvino.quantization import OpenVINOQuantization, OpenVINOQuantizationWithAccuracy


@Registry.register_dataset()
def cifar10_dataset(data_dir, **kwargs):
    import random

    from torch.utils.data import Subset
    from torchvision.datasets import CIFAR10
    from torchvision.transforms import ToTensor

    random.seed(1234)

    # define the full CIFAR10 test set
    full_test_set = CIFAR10(root=data_dir, train=False, transform=ToTensor(), download=True)

    # randomly sample n_test_samples from the full test set
    n_test_samples = 5
    random_indices = random.sample(range(len(full_test_set)), n_test_samples)
    return Subset(full_test_set, random_indices)


@pytest.fixture(name="cifar10_mv2_model", scope="module")
def cifar10_mv2_model_fixture(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("cifar10_mv2_model")
    torch_hub_model_path = "chenyaofo/pytorch-cifar-models"
    pytorch_hub_model_name = "cifar10_mobilenetv2_x1_0"
    torch.hub.set_dir(tmp_path)
    return PyTorchModelHandler(
        model_loader=lambda torch_hub_model_path: torch.hub.load(torch_hub_model_path, pytorch_hub_model_name),
        model_path=torch_hub_model_path,
        io_config={"input_names": ["input"], "input_shapes": [[1, 3, 32, 32]], "output_names": ["output"]},
    )


def test_openvino_quantization(tmp_path, cifar10_mv2_model):
    # setup
    ov_model = get_openvino_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        )
    }
    p = create_pass_from_dict(
        OpenVINOQuantization,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(ov_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert (Path(quantized_model.model_path) / "ov_model_quant.bin").is_file()
    assert (Path(quantized_model.model_path) / "ov_model_quant.xml").is_file()

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


def test_openvino_quantization_multi_ignore_scope(tmp_path, cifar10_mv2_model):
    # setup
    ov_model = get_openvino_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "ignored_scope": [["Add", "MatMul"], [".*Mul.*"]],
        "ignored_scope_type": ["types", "patterns"],
    }
    p = create_pass_from_dict(
        OpenVINOQuantization,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(ov_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert (Path(quantized_model.model_path) / "ov_model_quant.bin").is_file()
    assert (Path(quantized_model.model_path) / "ov_model_quant.xml").is_file()

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


def test_openvino_quantization_onnx_input(tmp_path, cifar10_mv2_model):
    # setup
    onnx_model = get_cifar10_mv2_onnx_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    def transform_to_np(data_item):
        image, _ = data_item
        return {"input": image.numpy()}

    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "transform_fn": transform_to_np,
    }
    p = create_pass_from_dict(
        OpenVINOQuantization,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(onnx_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()

    # cleanup
    shutil.rmtree(data_dir)
    if Path(quantized_model.model_path).is_file():
        q_dir = Path(quantized_model.model_path).parent
    else:
        q_dir = Path(quantized_model.model_path)
    shutil.rmtree(q_dir)


def test_openvino_quantization_onnx_input_multi_ignore_scope(tmp_path, cifar10_mv2_model):
    # setup
    onnx_model = get_cifar10_mv2_onnx_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    def transform_to_np(data_item):
        image, _ = data_item
        return {"input": image.numpy()}

    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "transform_fn": transform_to_np,
        "ignored_scope": [["Add"], [".*Add.*"]],
        "ignored_scope_type": ["types", "patterns"],
    }
    p = create_pass_from_dict(
        OpenVINOQuantization,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(onnx_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()

    # cleanup
    shutil.rmtree(data_dir)
    if Path(quantized_model.model_path).is_file():
        q_dir = Path(quantized_model.model_path).parent
    else:
        q_dir = Path(quantized_model.model_path)
    shutil.rmtree(q_dir)


def test_openvino_quantization_with_accuracy(tmp_path, cifar10_mv2_model):
    # setup
    ov_model = get_openvino_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        )
    }
    p = create_pass_from_dict(
        OpenVINOQuantizationWithAccuracy,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(ov_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert (Path(quantized_model.model_path) / "ov_model_quant.bin").is_file()
    assert (Path(quantized_model.model_path) / "ov_model_quant.xml").is_file()

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


def test_openvino_quantization_with_accuracy_multi_ignore_scope(tmp_path, cifar10_mv2_model):
    # setup
    ov_model = get_openvino_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "ignored_scope": [["Add", "MatMul"], [".*Mul.*"]],
        "ignored_scope_type": ["types", "patterns"],
    }
    p = create_pass_from_dict(
        OpenVINOQuantizationWithAccuracy,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(ov_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert (Path(quantized_model.model_path) / "ov_model_quant.bin").is_file()
    assert (Path(quantized_model.model_path) / "ov_model_quant.xml").is_file()

    # cleanup
    shutil.rmtree(quantized_model.model_path)
    shutil.rmtree(data_dir)


def test_openvino_quantization_with_accuracy_onnx_input(tmp_path, cifar10_mv2_model):
    import numpy as np

    # setup
    onnx_model = get_cifar10_mv2_onnx_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    def transform_to_np(data_item):
        image, _ = data_item
        return {"input": image.numpy()}

    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "transform_fn": transform_to_np,
        "max_drop": np.inf,
    }

    p = create_pass_from_dict(
        OpenVINOQuantizationWithAccuracy,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(onnx_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()

    # cleanup
    shutil.rmtree(data_dir)
    if Path(quantized_model.model_path).is_file():
        q_dir = Path(quantized_model.model_path).parent
    else:
        q_dir = Path(quantized_model.model_path)
    shutil.rmtree(q_dir)


def test_openvino_quantization_with_accuracy_onnx_input_multi_ignore_scope(tmp_path, cifar10_mv2_model):
    import numpy as np

    # setup
    onnx_model = get_cifar10_mv2_onnx_model(tmp_path, cifar10_mv2_model)
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)

    def transform_to_np(data_item):
        image, _ = data_item
        return {"input": image.numpy()}

    config = {
        "data_config": DataConfig(
            name="test_dc_config",
            load_dataset_config=DataComponentConfig(type="cifar10_dataset", params={"data_dir": str(data_dir)}),
            dataloader_config=DataComponentConfig(params={"shuffle": True}),
        ),
        "transform_fn": transform_to_np,
        "max_drop": np.inf,
        "ignored_scope": [["Add"], [".*Add.*"]],
        "ignored_scope_type": ["types", "patterns"],
    }

    p = create_pass_from_dict(
        OpenVINOQuantizationWithAccuracy,
        config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "quantized")

    # execute
    quantized_model = p.run(onnx_model, output_folder)

    # assert
    assert Path(quantized_model.model_path).exists()
    assert Path(quantized_model.model_path).is_file()

    # cleanup
    shutil.rmtree(data_dir)
    if Path(quantized_model.model_path).is_file():
        q_dir = Path(quantized_model.model_path).parent
    else:
        q_dir = Path(quantized_model.model_path)
    shutil.rmtree(q_dir)


def get_openvino_model(tmp_path, cifar10_mv2_model):
    openvino_conversion_config = {
        "input_shapes": [[1, 3, 32, 32]],
    }

    p = create_pass_from_dict(
        OpenVINOConversion,
        openvino_conversion_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    output_folder = str(tmp_path / "openvino")

    # remove any cached loaded model
    cifar10_mv2_model.model = None

    # execute
    return p.run(cifar10_mv2_model, output_folder)


def get_cifar10_mv2_onnx_model(tmp_path, cifar10_mv2_model):
    onnx_model_path = tmp_path / "cifar10_mobilenetv2.onnx"
    onnx_conversion_config = {"dynamic": False}
    p = create_pass_from_dict(
        OnnxConversion,
        onnx_conversion_config,
        disable_search=True,
        accelerator_spec=AcceleratorSpec("cpu", "OpenVINOExecutionProvider"),
    )
    cifar10_mv2_model.model = None
    return p.run(cifar10_mv2_model, str(onnx_model_path))
