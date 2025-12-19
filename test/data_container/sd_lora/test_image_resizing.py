# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# pylint: disable=redefined-outer-name
import pytest
from PIL import Image

from olive.data.component.sd_lora.dataset import ImageFolderDataset
from olive.data.component.sd_lora.image_resizing import image_resizing


@pytest.fixture
def temp_dataset(tmp_path):
    img = Image.new("RGB", (800, 600), color="red")
    img.save(tmp_path / "test.jpg")
    (tmp_path / "test.txt").write_text("caption")
    return ImageFolderDataset(data_dir=str(tmp_path))


def test_contain_mode(temp_dataset, tmp_path):
    output_dir = tmp_path / "output"
    image_resizing(temp_dataset, target_resolution=512, resize_mode="contain", output_dir=str(output_dir))
    output_img = Image.open(output_dir / "test.jpg")
    assert output_img.size == (512, 512)


def test_cover_mode(temp_dataset, tmp_path):
    output_dir = tmp_path / "output"
    image_resizing(temp_dataset, target_resolution=512, resize_mode="cover", output_dir=str(output_dir))
    output_img = Image.open(output_dir / "test.jpg")
    assert output_img.size == (512, 512)


def test_invalid_resize_mode(temp_dataset):
    with pytest.raises(ValueError, match="is not a valid ResizeMode"):
        image_resizing(temp_dataset, resize_mode="invalid")


def test_overwrite(temp_dataset, tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    existing = Image.new("RGB", (100, 100), color="blue")
    existing.save(output_dir / "test.jpg")

    # overwrite=False should skip
    image_resizing(
        temp_dataset, target_resolution=512, resize_mode="contain", output_dir=str(output_dir), overwrite=False
    )
    size = Image.open(output_dir / "test.jpg").size
    assert size == (100, 100)

    # overwrite=True should replace
    image_resizing(
        temp_dataset, target_resolution=512, resize_mode="contain", output_dir=str(output_dir), overwrite=True
    )
    size = Image.open(output_dir / "test.jpg").size
    assert size == (512, 512)
