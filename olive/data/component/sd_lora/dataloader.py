# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Dataloader for Stable Diffusion LoRA training with bucket batching."""

import logging
import random
from collections import defaultdict
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Sampler

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


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
        super().__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

        # Group indices by bucket
        self.bucket_indices = defaultdict(list)

        if hasattr(dataset, "bucket_assignments"):
            for i in range(len(dataset)):
                item = dataset[i]
                image_path = str(item.get("image_path", ""))
                if image_path in dataset.bucket_assignments:
                    bucket = tuple(dataset.bucket_assignments[image_path]["bucket"])
                    self.bucket_indices[bucket].append(i)
                else:
                    # Default bucket
                    self.bucket_indices[(512, 512)].append(i)
        else:
            # No bucket assignments, use single default bucket
            self.bucket_indices[(512, 512)] = list(range(len(dataset)))

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

        for bucket, indices in self.bucket_indices.items():
            # Shuffle indices within bucket
            if self.shuffle:
                indices = indices.copy()
                if g is not None:
                    perm = torch.randperm(len(indices), generator=g).tolist()
                    indices = [indices[i] for i in perm]
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

        for batch in all_batches:
            yield batch

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


def sd_lora_collate_fn(batch):
    """Collate function for SD LoRA batches.

    Handles batches where all images have the same dimensions (from the same bucket).
    """
    from PIL import Image

    import numpy as np

    # Separate components
    images = []
    captions = []
    metadata = []

    for item in batch:
        image_path = item.get("image_path")
        caption = item.get("caption", "")

        # Load image if path is provided
        if image_path and Path(image_path).exists():
            img = Image.open(image_path).convert("RGB")
            img_array = np.array(img, dtype=np.float32) / 255.0
            # Convert to CHW format
            img_array = np.transpose(img_array, (2, 0, 1))
            images.append(torch.from_numpy(img_array))

        captions.append(caption)

        # Collect other metadata
        meta = {k: v for k, v in item.items() if k not in ("image_path", "caption")}
        metadata.append(meta)

    # Stack images if they all have the same size
    if images:
        try:
            images = torch.stack(images)
        except RuntimeError:
            # Images have different sizes, keep as list
            pass

    return {
        "images": images,
        "captions": captions,
        "metadata": metadata,
    }


@Registry.register_dataloader()
@Registry.register_dataloader("image_bucket_dataloader")
def sd_lora_bucket_dataloader(
    dataset,
    batch_size: int = 1,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    seed: Optional[int] = None,
    pin_memory: bool = True,
    persistent_workers: bool = False,
    prefetch_factor: Optional[int] = None,
    **kwargs,
):
    """Create a dataloader for SD LoRA training with bucket batching.

    This dataloader groups images by their assigned buckets, ensuring
    that all images in a batch have the same dimensions.

    Args:
        dataset: The SD LoRA dataset with bucket assignments.
        batch_size: Number of samples per batch.
        shuffle: Whether to shuffle samples.
        num_workers: Number of worker processes for data loading.
        drop_last: Whether to drop the last incomplete batch.
        seed: Random seed for reproducibility.
        pin_memory: Whether to pin memory for faster GPU transfer.
        persistent_workers: Whether to keep workers alive between epochs.
        prefetch_factor: Number of batches to prefetch per worker.
        **kwargs: Additional DataLoader arguments.

    Returns:
        DataLoader instance with bucket batch sampling.
    """
    # Create bucket batch sampler
    batch_sampler = BucketBatchSampler(
        dataset,
        batch_size=batch_size,
        drop_last=drop_last,
        shuffle=shuffle,
        seed=seed,
    )

    # Build dataloader kwargs
    loader_kwargs = {
        "batch_sampler": batch_sampler,
        "num_workers": num_workers,
        "collate_fn": sd_lora_collate_fn,
        "pin_memory": pin_memory and torch.cuda.is_available(),
    }

    if num_workers > 0:
        loader_kwargs["persistent_workers"] = persistent_workers
        if prefetch_factor is not None:
            loader_kwargs["prefetch_factor"] = prefetch_factor

    # Remove conflicting kwargs
    kwargs.pop("batch_size", None)
    kwargs.pop("shuffle", None)
    kwargs.pop("drop_last", None)
    kwargs.pop("sampler", None)

    loader_kwargs.update(kwargs)

    return DataLoader(dataset, **loader_kwargs)
