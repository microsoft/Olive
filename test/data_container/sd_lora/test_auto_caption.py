# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: SLF001
# pylint: disable=protected-access
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

import olive.data.component.sd_lora.auto_caption as auto_caption_module
from olive.data.component.sd_lora.dataset import ImageFolderDataset


def test_load_image(tmp_path):
    img = Image.new("RGBA", (512, 512), color=(255, 0, 0, 128))
    img.save(tmp_path / "test.png")
    loaded = auto_caption_module._load_image(str(tmp_path / "test.png"))
    assert loaded.mode == "RGB"


def test_save_caption_to_file(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.touch()
    auto_caption_module._save_caption(str(image_path), "test caption")
    assert (tmp_path / "test.txt").read_text() == "test caption"


def test_save_caption_to_memory(tmp_path):
    mock_dataset = MagicMock()
    image_path = tmp_path / "test.jpg"
    auto_caption_module._save_caption(str(image_path), "test caption", dataset=mock_dataset, index=0)
    mock_dataset.set_caption.assert_called_once_with(0, "test caption")


def test_auto_caption_invalid_model_type(tmp_path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(tmp_path / "test.jpg")
    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    with pytest.raises(ValueError, match="Unsupported model type"):
        auto_caption_module.auto_caption(dataset, model_type="invalid")


def test_auto_caption_blip2_dispatch(tmp_path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(tmp_path / "test.jpg")
    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    with patch.object(auto_caption_module, "blip2_caption") as mock_blip2:
        mock_blip2.return_value = dataset
        auto_caption_module.auto_caption(dataset, model_type="blip2")
        mock_blip2.assert_called_once()


def test_auto_caption_florence2_dispatch(tmp_path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(tmp_path / "test.jpg")
    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    with patch.object(auto_caption_module, "florence2_caption") as mock_florence2:
        mock_florence2.return_value = dataset
        auto_caption_module.auto_caption(dataset, model_type="florence2")
        mock_florence2.assert_called_once()


def test_auto_caption_trigger_word(tmp_path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(tmp_path / "test.jpg")
    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    with patch.object(auto_caption_module, "blip2_caption") as mock_blip2:
        mock_blip2.return_value = dataset
        auto_caption_module.auto_caption(dataset, model_type="blip2", trigger_word="sks")
        call_kwargs = mock_blip2.call_args[1]
        assert call_kwargs["prefix"] == "sks, "
