# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import logging
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def image_filtering(
    dataset,
    min_size: int = 256,
    min_aspect_ratio: float = 0.5,
    max_aspect_ratio: float = 2.0,
    remove_duplicates: bool = False,
    blur_threshold: Optional[float] = None,
):
    """Filter images based on size and quality criteria.

    Args:
        dataset: The dataset to process.
        min_size: Minimum width and height in pixels.
        min_aspect_ratio: Minimum aspect ratio (width/height).
        max_aspect_ratio: Maximum aspect ratio (width/height).
        remove_duplicates: Whether to detect and remove duplicate images.
        blur_threshold: Minimum blur score (higher = sharper). Images below this are filtered.

    Returns:
        The filtered dataset.

    """
    from PIL import Image

    filtered_indices = []
    reason_counts = {}

    # Track duplicates if enabled
    image_hashes: set[str] = set()

    def filter_image(idx, reason):
        filtered_indices.append(idx)
        reason_counts[reason] = reason_counts.get(reason, 0) + 1

    for i, item in enumerate(dataset):
        image_path = Path(item["image_path"])

        # Check file existence
        if not image_path.exists():
            filter_image(i, "file_not_found")
            continue

        # Load image for dimension checks
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height
        except Exception:
            filter_image(i, "load_error")
            continue

        # Check dimensions
        if width < min_size or height < min_size:
            filter_image(i, "too_small")
            continue

        # Check aspect ratio
        if aspect_ratio < min_aspect_ratio or aspect_ratio > max_aspect_ratio:
            filter_image(i, "bad_aspect_ratio")
            continue

        # Check for duplicates using perceptual hashing
        if remove_duplicates:
            img_hash = _compute_image_hash(image_path)
            if img_hash:
                if img_hash in image_hashes:
                    filter_image(i, "duplicate")
                    continue
                image_hashes.add(img_hash)

        # Check blur level
        if blur_threshold is not None:
            blur_score = _compute_blur_score(image_path)
            if blur_score < blur_threshold:
                filter_image(i, "too_blurry")
                continue

    # Log filtering results
    logger.info(
        "Filtering: %d / %d images passed (%d filtered)",
        len(dataset) - len(filtered_indices),
        len(dataset),
        len(filtered_indices),
    )

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        logger.info("  - %s: %d", reason, count)

    # Update dataset by removing filtered images
    filtered_set = set(filtered_indices)
    dataset.image_paths = [p for i, p in enumerate(dataset.image_paths) if i not in filtered_set]

    return dataset


def _compute_image_hash(image_path: Path, hash_size: int = 16) -> Optional[str]:
    """Compute perceptual hash for an image using average hashing."""
    try:
        import numpy as np
        from PIL import Image

        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            processed = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)

            # Compute average
            pixels = np.array(processed)
            avg = pixels.mean()

            # Create hash
            hash_bits = (pixels > avg).flatten()
            return "".join("1" if b else "0" for b in hash_bits)
    except Exception:
        return None


def _compute_blur_score(image_path: Path) -> float:
    """Compute blur score using Laplacian variance (higher = sharper)."""
    try:
        import cv2

        img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return 0.0

        # Resize for consistent comparison
        img = cv2.resize(img, (512, 512))

        # Compute Laplacian variance
        laplacian = cv2.Laplacian(img, cv2.CV_64F)
        variance = laplacian.var()

        return float(variance)
    except Exception:
        return 0.0
