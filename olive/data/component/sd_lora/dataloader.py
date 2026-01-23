# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import random
from collections import defaultdict
from typing import Optional

import torch
from torch.utils.data import Sampler


class BucketBatchSampler(Sampler):
    """Sampler that batches images from the same bucket together.

    This ensures that all images in a batch have the same dimensions,
    which is required for efficient training without excessive padding.
    """

    def __init__(
        self,
        dataset,
        batch_size: int = 1,
        drop_last: bool = False,
        shuffle: bool = True,
        seed: Optional[int] = None,
    ):
        """Initialize the bucket batch sampler.

        Args:
            dataset: The dataset with bucket_assignments attribute.
            batch_size: Number of samples per batch.
            drop_last: Whether to drop the last incomplete batch.
            shuffle: Whether to shuffle samples within and across buckets.
            seed: Random seed for reproducibility.

        """
        super().__init__()
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by bucket
        if not getattr(dataset, "bucket_assignments", None):
            raise ValueError("Dataset must have bucket_assignments. Run aspect_ratio_bucketing first.")

        self.bucket_indices = defaultdict(list)
        for i, item in enumerate(dataset):
            image_path = str(item.get("image_path", ""))
            bucket = tuple(dataset.bucket_assignments[image_path]["bucket"])
            self.bucket_indices[bucket].append(i)

    def __iter__(self):
        """Generate batches grouped by bucket."""
        # Set random seed for this epoch
        if self.seed is not None:
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
        else:
            g = None

        # Create list of all batches
        all_batches = []

        for indices in self.bucket_indices.values():
            # Shuffle indices within bucket
            if self.shuffle:
                indices = indices.copy()  # noqa: PLW2901
                if g is not None:
                    perm = torch.randperm(len(indices), generator=g).tolist()
                    indices = [indices[i] for i in perm]  # noqa: PLW2901
                else:
                    random.shuffle(indices)

            # Create batches
            for i in range(0, len(indices), self.batch_size):
                batch = indices[i : i + self.batch_size]
                if len(batch) < self.batch_size and self.drop_last:
                    continue
                all_batches.append(batch)

        # Shuffle batches across buckets
        if self.shuffle:
            if g is not None:
                perm = torch.randperm(len(all_batches), generator=g).tolist()
                all_batches = [all_batches[i] for i in perm]
            else:
                random.shuffle(all_batches)

        yield from all_batches

    def __len__(self):
        """Return the number of batches."""
        total_batches = 0
        for indices in self.bucket_indices.values():
            n_batches = len(indices) // self.batch_size
            if not self.drop_last and len(indices) % self.batch_size != 0:
                n_batches += 1
            total_batches += n_batches
        return total_batches

    def set_epoch(self, epoch: int):
        """Set the epoch for deterministic shuffling."""
        self.epoch = epoch
