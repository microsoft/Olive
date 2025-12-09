# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
from pathlib import Path

import pytest
from PIL import Image

from olive.data.component.sd_lora.aspect_ratio_bucketing import (
    SD15_BUCKETS,
    SDXL_BUCKETS,
    aspect_ratio_bucketing,
    generate_buckets,
)
from olive.data.component.sd_lora.dataset import ImageFolderDataset


class TestGenerateBuckets:
    def test_default_buckets(self):
        buckets = generate_buckets(base_resolution=512)
        assert len(buckets) > 0
        assert any(w == h for w, h in buckets)

    def test_divisibility(self):
        buckets = generate_buckets(base_resolution=512, divisor=64)
        for w, h in buckets:
            assert w % 64 == 0
            assert h % 64 == 0


class TestAspectRatioBucketing:
    @pytest.fixture
    def temp_dataset(self, tmp_path):
        sizes = [(512, 512), (800, 600), (600, 800)]
        for i, (w, h) in enumerate(sizes):
            img = Image.new("RGB", (w, h), color=(i * 50, i * 50, i * 50))
            img.save(tmp_path / f"image_{i}.jpg")
            (tmp_path / f"image_{i}.txt").write_text(f"Caption {i}")

        return ImageFolderDataset(data_dir=str(tmp_path))

    def test_sd15_buckets(self, temp_dataset, tmp_path):
        output_dir = tmp_path / "output"
        result = aspect_ratio_bucketing(temp_dataset, base_resolution=512, output_dir=str(output_dir))
        assert hasattr(result, "bucket_assignments")
        assert hasattr(result, "buckets")
        assert result.buckets == SD15_BUCKETS

    def test_sdxl_buckets(self, temp_dataset, tmp_path):
        output_dir = tmp_path / "output"
        result = aspect_ratio_bucketing(temp_dataset, base_resolution=1024, output_dir=str(output_dir))
        assert result.buckets == SDXL_BUCKETS

    def test_custom_buckets(self, temp_dataset, tmp_path):
        output_dir = tmp_path / "output"
        custom = [(512, 512), (640, 480)]
        result = aspect_ratio_bucketing(temp_dataset, custom_buckets=custom, output_dir=str(output_dir))
        assert result.buckets == custom

    def test_bucket_assignments(self, temp_dataset, tmp_path):
        output_dir = tmp_path / "output"
        result = aspect_ratio_bucketing(temp_dataset, base_resolution=512, output_dir=str(output_dir))
        assert len(result.bucket_assignments) == len(temp_dataset)

    def test_resize_images(self, temp_dataset, tmp_path):
        output_dir = tmp_path / "output"
        aspect_ratio_bucketing(
            temp_dataset,
            base_resolution=512,
            resize_images=True,
            output_dir=str(output_dir),
        )
        output_files = list(Path(output_dir).glob("*.jpg"))
        assert len(output_files) == len(temp_dataset)
