# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
# ruff: noqa: SLF001
# pylint: disable=protected-access
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest
from PIL import Image

from olive.data.component.sd_lora.dataset import (
    HuggingFaceImageDataset,
    ImageFolderDataset,
    image_folder_dataset,
)


class TestImageFolderDataset:
    @pytest.fixture
    def temp_image_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            for i in range(3):
                img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
                img_path = Path(tmpdir) / f"image_{i}.jpg"
                img.save(img_path)

                caption_path = Path(tmpdir) / f"image_{i}.txt"
                caption_path.write_text(f"Caption for image {i}")

            yield tmpdir

    def test_load_images(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir)
        assert len(dataset) == 3

    def test_get_item(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir)
        item = dataset[0]
        assert "image_path" in item
        assert "caption" in item

    def test_load_caption(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir)
        item = dataset[0]
        assert "Caption for image" in item["caption"]

    def test_default_caption(self, temp_image_dir):
        for f in Path(temp_image_dir).glob("*.txt"):
            f.unlink()

        dataset = ImageFolderDataset(data_dir=temp_image_dir, default_caption="default")
        item = dataset[0]
        assert item["caption"] == "default"

    def test_instance_prompt(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir, instance_prompt="a photo of sks dog")
        item = dataset[0]
        assert item["caption"] == "a photo of sks dog"

    def test_max_samples(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir, max_samples=2)
        assert len(dataset) == 2

    def test_get_all_image_paths(self, temp_image_dir):
        dataset = ImageFolderDataset(data_dir=temp_image_dir)
        paths = dataset.get_all_image_paths()
        assert len(paths) == 3
        assert all(isinstance(p, str) for p in paths)

    def test_recursive_search(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        img = Image.new("RGB", (512, 512), color="red")
        img.save(subdir / "test.jpg")

        dataset = ImageFolderDataset(data_dir=str(tmp_path), recursive=True)
        assert len(dataset) == 1

    def test_non_recursive_search(self, tmp_path):
        subdir = tmp_path / "subdir"
        subdir.mkdir()
        img = Image.new("RGB", (512, 512), color="red")
        img.save(subdir / "test.jpg")

        dataset = ImageFolderDataset(data_dir=str(tmp_path), recursive=False)
        assert len(dataset) == 0


class TestImageFolderDatasetFunction:
    def test_registered_function(self, tmp_path):
        img = Image.new("RGB", (512, 512), color="red")
        img.save(tmp_path / "test.jpg")
        (tmp_path / "test.txt").write_text("test caption")

        dataset = image_folder_dataset(data_dir=str(tmp_path))
        assert len(dataset) == 1
        assert dataset[0]["caption"] == "test caption"


class TestHuggingFaceImageDataset:
    @pytest.fixture
    def mock_hf_dataset(self, tmp_path):
        images = []
        for i in range(3):
            img = Image.new("RGB", (512, 512), color=(i * 50, i * 50, i * 50))
            img_path = tmp_path / f"image_{i}.jpg"
            img.save(img_path)
            mock_img = MagicMock()
            mock_img.filename = str(img_path)
            images.append(mock_img)

        mock_ds = MagicMock()
        mock_ds.column_names = ["image", "caption"]
        mock_ds.__len__ = MagicMock(return_value=3)
        mock_ds.__getitem__ = MagicMock(
            side_effect=lambda i: {"image": images[i], "caption": f"Caption {i}"} if isinstance(i, int) else None
        )
        return mock_ds

    def test_init(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="caption")
        assert len(dataset) == 3

    def test_get_item(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="caption")
        item = dataset[0]
        assert "image_path" in item
        assert "caption" in item
        assert item["caption"] == "Caption 0"

    def test_set_caption(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="caption")
        dataset.set_caption(0, "New caption")
        item = dataset[0]
        assert item["caption"] == "New caption"

    def test_get_caption_priority(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="caption")
        # Set caption in memory
        dataset.set_caption(0, "Memory caption")
        # Memory caption should take priority
        assert dataset.get_caption(0) == "Memory caption"

    def test_invalid_image_column(self, mock_hf_dataset):
        with pytest.raises(ValueError, match="Image column"):
            HuggingFaceImageDataset(mock_hf_dataset, image_column="invalid", caption_column="caption")

    def test_invalid_caption_column(self, mock_hf_dataset):
        with pytest.raises(ValueError, match="Caption column"):
            HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="invalid")

    def test_no_caption_column(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column=None)
        assert dataset.get_caption(0) == ""

    def test_set_image_path(self, mock_hf_dataset):
        dataset = HuggingFaceImageDataset(mock_hf_dataset, image_column="image", caption_column="caption")
        dataset.set_image_path(0, "/new/path/image.jpg")
        assert dataset._image_paths_cache[0] == "/new/path/image.jpg"
