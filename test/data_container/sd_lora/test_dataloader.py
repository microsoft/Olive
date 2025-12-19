# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import pytest
from PIL import Image

from olive.data.component.sd_lora.dataloader import BucketBatchSampler
from olive.data.component.sd_lora.dataset import ImageFolderDataset


class TestBucketBatchSampler:
    @pytest.fixture
    def dataset_with_buckets(self, tmp_path):
        for i in range(6):
            img = Image.new("RGB", (512, 512), color=(i * 40, i * 40, i * 40))
            img.save(tmp_path / f"image_{i}.jpg")
            (tmp_path / f"image_{i}.txt").write_text(f"Caption {i}")

        dataset = ImageFolderDataset(data_dir=str(tmp_path))

        # Manually set bucket assignments
        dataset.bucket_assignments = {}
        for i, item in enumerate(dataset):
            if i < 3:
                bucket = (512, 512)
            else:
                bucket = (576, 448)
            dataset.bucket_assignments[item["image_path"]] = {"bucket": bucket}

        return dataset

    def test_init(self, dataset_with_buckets):
        sampler = BucketBatchSampler(dataset_with_buckets, batch_size=2)
        assert sampler.batch_size == 2
        assert len(sampler.bucket_indices) == 2

    def test_iter(self, dataset_with_buckets):
        sampler = BucketBatchSampler(dataset_with_buckets, batch_size=2, shuffle=False)
        batches = list(sampler)
        assert len(batches) > 0
        # Each batch should have at most batch_size items
        for batch in batches:
            assert len(batch) <= 2

    def test_len(self, dataset_with_buckets):
        sampler = BucketBatchSampler(dataset_with_buckets, batch_size=2, drop_last=False)
        expected_batches = 2 + 2  # ceil(3/2) + ceil(3/2)
        assert len(sampler) == expected_batches

    def test_drop_last(self, dataset_with_buckets):
        sampler = BucketBatchSampler(dataset_with_buckets, batch_size=2, drop_last=True)
        batches = list(sampler)
        for batch in batches:
            assert len(batch) == 2

    def test_set_epoch(self, dataset_with_buckets):
        sampler = BucketBatchSampler(dataset_with_buckets, batch_size=2, seed=42)
        sampler.set_epoch(1)
        assert sampler.epoch == 1

    def test_deterministic_with_seed(self, dataset_with_buckets):
        sampler1 = BucketBatchSampler(dataset_with_buckets, batch_size=2, seed=42)
        sampler2 = BucketBatchSampler(dataset_with_buckets, batch_size=2, seed=42)

        batches1 = list(sampler1)
        batches2 = list(sampler2)

        assert batches1 == batches2

    def test_no_bucket_assignments(self, tmp_path):
        img = Image.new("RGB", (512, 512), color="red")
        img.save(tmp_path / "test.jpg")
        (tmp_path / "test.txt").write_text("caption")

        dataset = ImageFolderDataset(data_dir=str(tmp_path))
        # No bucket_assignments set

        with pytest.raises(ValueError, match="must have bucket_assignments"):
            BucketBatchSampler(dataset, batch_size=1)
