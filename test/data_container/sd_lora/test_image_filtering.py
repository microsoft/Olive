# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: PLW0621
# pylint: disable=redefined-outer-name
import numpy as np
import pytest
from PIL import Image

from olive.data.component.sd_lora.dataset import ImageFolderDataset
from olive.data.component.sd_lora.image_filtering import image_filtering


@pytest.fixture
def temp_dataset(tmp_path):
    # Create images of different sizes
    sizes = [(512, 512), (200, 200), (1024, 1024), (800, 400)]
    for i, (w, h) in enumerate(sizes):
        img = Image.new("RGB", (w, h), color=(i * 60, i * 60, i * 60))
        img.save(tmp_path / f"image_{i}.jpg")
        (tmp_path / f"image_{i}.txt").write_text(f"Caption {i}")

    return ImageFolderDataset(data_dir=str(tmp_path))


def test_filter_by_min_size(temp_dataset):
    original_len = len(temp_dataset)
    result = image_filtering(temp_dataset, min_size=400)
    # 200x200 image should be filtered
    assert len(result.image_paths) < original_len


def test_filter_by_aspect_ratio(temp_dataset):
    original_len = len(temp_dataset)
    result = image_filtering(temp_dataset, max_aspect_ratio=1.5)
    # 800x400 (aspect=2.0) should be filtered
    assert len(result.image_paths) < original_len


def test_remove_duplicates(tmp_path):
    # Create two identical images
    arr = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        arr[:, i, :] = i // 2
    img = Image.fromarray(arr)
    img.save(tmp_path / "img1.jpg")
    img.save(tmp_path / "img2.jpg")  # duplicate

    # Create one different image
    arr2 = np.zeros((512, 512, 3), dtype=np.uint8)
    for i in range(512):
        arr2[i, :, :] = i // 2
    Image.fromarray(arr2).save(tmp_path / "img3.jpg")

    dataset = ImageFolderDataset(data_dir=str(tmp_path))
    result = image_filtering(dataset, remove_duplicates=True)
    assert len(result.image_paths) == 2
