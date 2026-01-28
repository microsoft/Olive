# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from unittest.mock import MagicMock

import pytest
import torch

from olive.model import DiffusersModelHandler


@pytest.fixture
def test_image_folder(tmp_path):
    """Create a test image folder with images and captions."""
    from PIL import Image

    data_dir = tmp_path / "train_images"
    data_dir.mkdir(parents=True, exist_ok=True)

    for i in range(4):
        img = Image.new("RGB", (64, 64), color=(i * 50, i * 50, i * 50))
        img.save(data_dir / f"image_{i}.png")
        (data_dir / f"image_{i}.txt").write_text(f"a test image {i}")

    return str(data_dir)


@pytest.fixture
def output_folder(tmp_path):
    """Create output folder."""
    folder = tmp_path / "output"
    folder.mkdir()
    return str(folder)


@pytest.fixture
def mock_accelerator():
    """Create a mock accelerator."""
    mock_acc = MagicMock()
    mock_acc.device = "cpu"
    mock_acc.mixed_precision = "no"
    mock_acc.num_processes = 1
    mock_acc.is_main_process = True
    mock_acc.is_local_main_process = True
    mock_acc.sync_gradients = True
    mock_acc.gradient_accumulation_steps = 1
    mock_acc.prepare.side_effect = lambda *args: args
    mock_acc.backward = MagicMock()
    mock_acc.clip_grad_norm_ = MagicMock()
    mock_acc.unwrap_model = lambda x: x
    mock_acc.wait_for_everyone = MagicMock()
    mock_acc.end_training = MagicMock()
    mock_acc.save_state = MagicMock()
    mock_acc.accumulate = MagicMock(return_value=MagicMock(__enter__=MagicMock(), __exit__=MagicMock()))
    return mock_acc


@pytest.fixture
def mock_torch_model():
    """Create a mock torch model with parameters."""
    mock_model = MagicMock()
    param = torch.nn.Parameter(torch.randn(4, 4))
    mock_model.parameters.return_value = [param]
    mock_model.named_parameters.return_value = [("weight", param)]
    mock_model.requires_grad_ = MagicMock(return_value=mock_model)
    mock_model.to = MagicMock(return_value=mock_model)
    mock_model.train = MagicMock(return_value=mock_model)
    mock_model.eval = MagicMock(return_value=mock_model)
    mock_model.config = MagicMock()
    mock_model.config.in_channels = 4
    return mock_model


@pytest.fixture
def mock_input_model_sd15():
    """Create mock input model for SD 1.5."""
    model = MagicMock(spec=DiffusersModelHandler)
    model.model_path = "runwayml/stable-diffusion-v1-5"
    model.model_attributes = {}
    model.get_resource.return_value = model.model_path
    return model


@pytest.fixture
def mock_input_model_sdxl():
    """Create mock input model for SDXL."""
    model = MagicMock(spec=DiffusersModelHandler)
    model.model_path = "stabilityai/stable-diffusion-xl-base-1.0"
    model.model_attributes = {}
    model.get_resource.return_value = model.model_path
    return model


@pytest.fixture
def mock_input_model_flux():
    """Create mock input model for Flux."""
    model = MagicMock(spec=DiffusersModelHandler)
    model.model_path = "black-forest-labs/FLUX.1-dev"
    model.model_attributes = {}
    model.get_resource.return_value = model.model_path
    return model
