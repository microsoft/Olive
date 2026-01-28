# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: SLF001
# pylint: disable=protected-access
from unittest.mock import patch

from PIL import Image

import olive.data.component.sd_lora.auto_tagging as auto_tagging_module
from olive.data.component.sd_lora.dataset import ImageFolderDataset


def test_save_tags(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.touch()
    auto_tagging_module._save_tags(str(image_path), ["tag1", "tag2", "tag3"])
    assert (tmp_path / "test.txt").read_text() == "tag1, tag2, tag3"


def test_save_empty_tags(tmp_path):
    image_path = tmp_path / "test.jpg"
    image_path.touch()
    auto_tagging_module._save_tags(str(image_path), [])
    assert (tmp_path / "test.txt").read_text() == ""


def test_auto_tagging_dispatch(tmp_path):
    img = Image.new("RGB", (512, 512), color="red")
    img.save(tmp_path / "test.jpg")
    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    with patch.object(auto_tagging_module, "wd14_tagging") as mock_wd14:
        mock_wd14.return_value = dataset
        auto_tagging_module.auto_tagging(dataset)
        mock_wd14.assert_called_once()
