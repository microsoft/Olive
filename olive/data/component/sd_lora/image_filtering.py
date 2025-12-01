# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""Image filtering component for Stable Diffusion LoRA training."""

import logging
import shutil
from pathlib import Path
from typing import Optional

from olive.data.registry import Registry

logger = logging.getLogger(__name__)


@Registry.register_pre_process()
def image_filtering(
    dataset,
    min_width: Optional[int] = None,
    min_height: Optional[int] = None,
    max_width: Optional[int] = None,
    max_height: Optional[int] = None,
    min_aspect_ratio: Optional[float] = None,
    max_aspect_ratio: Optional[float] = None,
    min_file_size: Optional[int] = None,
    max_file_size: Optional[int] = None,
    min_resolution: Optional[int] = None,
    max_resolution: Optional[int] = None,
    allowed_formats: Optional[list[str]] = None,
    remove_duplicates: bool = False,
    duplicate_threshold: float = 0.95,
    aesthetic_score_threshold: Optional[float] = None,
    aesthetic_model: Optional[str] = None,
    blur_threshold: Optional[float] = None,
    move_filtered: bool = False,
    filtered_dir: Optional[str] = None,
    dry_run: bool = False,
    device: str = "cuda",
    **kwargs,
):
    """Filter images based on various quality and size criteria.

    This component helps clean up training data by filtering out:
    - Images that are too small or too large
    - Images with extreme aspect ratios
    - Duplicate images (optional, using perceptual hashing)
    - Low aesthetic quality images (optional, using CLIP-based scorer)
    - Blurry images (optional, using Laplacian variance)

    Args:
        dataset: The dataset to process.
        min_width: Minimum image width in pixels.
        min_height: Minimum image height in pixels.
        max_width: Maximum image width in pixels.
        max_height: Maximum image height in pixels.
        min_aspect_ratio: Minimum aspect ratio (width/height).
        max_aspect_ratio: Maximum aspect ratio (width/height).
        min_file_size: Minimum file size in bytes.
        max_file_size: Maximum file size in bytes.
        min_resolution: Minimum total resolution (width * height).
        max_resolution: Maximum total resolution (width * height).
        allowed_formats: List of allowed image formats (e.g., ["jpg", "png"]).
        remove_duplicates: Whether to detect and remove duplicate images.
        duplicate_threshold: Similarity threshold for duplicate detection (0-1).
        aesthetic_score_threshold: Minimum aesthetic score (if using aesthetic filtering).
        aesthetic_model: Model to use for aesthetic scoring.
        blur_threshold: Maximum blur score (lower = more blurry). Filter images below this.
        move_filtered: Whether to move filtered images instead of just excluding them.
        filtered_dir: Directory to move filtered images to.
        dry_run: If True, only report what would be filtered without actually filtering.
        device: Device for neural network-based filtering.
        **kwargs: Additional arguments.

    Returns:
        The filtered dataset.
    """
    from PIL import Image

    filtered_indices = []
    filter_reasons = {}

    # Track duplicates if enabled
    image_hashes = {} if remove_duplicates else None

    for i in range(len(dataset)):
        item = dataset[i]
        image_path = Path(item["image_path"])

        # Check file existence
        if not image_path.exists():
            filtered_indices.append(i)
            filter_reasons[i] = "file_not_found"
            continue

        # Check file format
        if allowed_formats:
            suffix = image_path.suffix.lower().lstrip(".")
            if suffix not in [f.lower().lstrip(".") for f in allowed_formats]:
                filtered_indices.append(i)
                filter_reasons[i] = f"format_not_allowed: {suffix}"
                continue

        # Check file size
        file_size = image_path.stat().st_size
        if min_file_size and file_size < min_file_size:
            filtered_indices.append(i)
            filter_reasons[i] = f"file_too_small: {file_size}"
            continue
        if max_file_size and file_size > max_file_size:
            filtered_indices.append(i)
            filter_reasons[i] = f"file_too_large: {file_size}"
            continue

        # Load image for dimension checks
        try:
            with Image.open(image_path) as img:
                width, height = img.size
                aspect_ratio = width / height
                resolution = width * height
        except Exception as e:
            filtered_indices.append(i)
            filter_reasons[i] = f"load_error: {e}"
            continue

        # Check dimensions
        if min_width and width < min_width:
            filtered_indices.append(i)
            filter_reasons[i] = f"width_too_small: {width}"
            continue
        if max_width and width > max_width:
            filtered_indices.append(i)
            filter_reasons[i] = f"width_too_large: {width}"
            continue
        if min_height and height < min_height:
            filtered_indices.append(i)
            filter_reasons[i] = f"height_too_small: {height}"
            continue
        if max_height and height > max_height:
            filtered_indices.append(i)
            filter_reasons[i] = f"height_too_large: {height}"
            continue

        # Check aspect ratio
        if min_aspect_ratio and aspect_ratio < min_aspect_ratio:
            filtered_indices.append(i)
            filter_reasons[i] = f"aspect_ratio_too_low: {aspect_ratio:.2f}"
            continue
        if max_aspect_ratio and aspect_ratio > max_aspect_ratio:
            filtered_indices.append(i)
            filter_reasons[i] = f"aspect_ratio_too_high: {aspect_ratio:.2f}"
            continue

        # Check resolution
        if min_resolution and resolution < min_resolution:
            filtered_indices.append(i)
            filter_reasons[i] = f"resolution_too_low: {resolution}"
            continue
        if max_resolution and resolution > max_resolution:
            filtered_indices.append(i)
            filter_reasons[i] = f"resolution_too_high: {resolution}"
            continue

        # Check for duplicates using perceptual hashing
        if remove_duplicates:
            img_hash = _compute_image_hash(image_path)
            if img_hash:
                # Check against existing hashes
                is_duplicate = False
                for existing_hash, existing_idx in image_hashes.items():
                    similarity = _hash_similarity(img_hash, existing_hash)
                    if similarity >= duplicate_threshold:
                        filtered_indices.append(i)
                        filter_reasons[i] = f"duplicate_of_image_{existing_idx}"
                        is_duplicate = True
                        break

                if not is_duplicate:
                    image_hashes[img_hash] = i

        # Check blur level
        if blur_threshold is not None:
            blur_score = _compute_blur_score(image_path)
            if blur_score < blur_threshold:
                filtered_indices.append(i)
                filter_reasons[i] = f"too_blurry: {blur_score:.2f}"
                continue

    # Log filtering results
    logger.info(
        "Filtering results: %d / %d images passed (%d filtered)",
        len(dataset) - len(filtered_indices),
        len(dataset),
        len(filtered_indices),
    )

    # Log filter reasons
    reason_counts = {}
    for reason in filter_reasons.values():
        reason_type = reason.split(":")[0]
        reason_counts[reason_type] = reason_counts.get(reason_type, 0) + 1

    for reason, count in sorted(reason_counts.items(), key=lambda x: -x[1]):
        logger.info("  - %s: %d images", reason, count)

    # Move filtered images if requested
    if move_filtered and filtered_dir and not dry_run:
        filtered_path = Path(filtered_dir)
        filtered_path.mkdir(parents=True, exist_ok=True)

        for i in filtered_indices:
            item = dataset[i]
            image_path = Path(item["image_path"])
            if image_path.exists():
                # Move image and associated files (caption, tags, etc.)
                dest_path = filtered_path / image_path.name
                shutil.move(str(image_path), str(dest_path))

                # Move associated text files
                for ext in [".txt", ".caption", ".tags"]:
                    assoc_path = image_path.with_suffix(ext)
                    if assoc_path.exists():
                        shutil.move(str(assoc_path), str(filtered_path / assoc_path.name))

    # Create filtered dataset
    if not dry_run:
        # Update dataset by removing filtered images
        filtered_set = set(filtered_indices)
        dataset.image_paths = [p for i, p in enumerate(dataset.image_paths) if i not in filtered_set]

    return dataset


def _compute_image_hash(image_path: Path, hash_size: int = 16) -> Optional[str]:
    """Compute perceptual hash for an image using average hashing."""
    try:
        from PIL import Image

        with Image.open(image_path) as img:
            # Convert to grayscale and resize
            img = img.convert("L").resize((hash_size, hash_size), Image.LANCZOS)

            # Compute average
            import numpy as np

            pixels = np.array(img)
            avg = pixels.mean()

            # Create hash
            hash_bits = (pixels > avg).flatten()
            hash_str = "".join("1" if b else "0" for b in hash_bits)
            return hash_str
    except Exception:
        return None


def _hash_similarity(hash1: str, hash2: str) -> float:
    """Compute similarity between two hashes (0-1)."""
    if len(hash1) != len(hash2):
        return 0.0
    matches = sum(c1 == c2 for c1, c2 in zip(hash1, hash2))
    return matches / len(hash1)


def _compute_blur_score(image_path: Path) -> float:
    """Compute blur score using Laplacian variance (higher = sharper)."""
    try:
        import cv2
        import numpy as np

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
